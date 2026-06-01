import json
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from app.graph.state import DiscussionState, ChatMessage
from app.models import get_llm
from app.guardrails import parse_json_safely

FAITHFULNESS_SYSTEM_PROMPT = """You are an expert fact-checker evaluating the groundedness of an AI Tutor's statement.
Your job is to analyze the Tutor's Statement against the provided Search Context.

Identify each assertion made in the Tutor's Statement. Verify if it is directly supported or entailed by the Search Context.
If any factual claim in the Tutor's Statement is unsupported, missing, or contradicts the Search Context, classify it as UNFAITHFUL.
If the claims are fully supported or are benign conversational statements (like greetings, instructions, or simple transitions), classify as FAITHFUL.

You must respond in strict JSON format with exactly these two keys:
{
  "faithful": boolean,
  "reason": "empty string if faithful, or a detailed description of the unsupported/hallucinated claims"
}
Do not return any other text besides the JSON."""

async def check_tutor_faithfulness(statement: str, search_context: str, mode: str, model_choice: str) -> dict:
    try:
        # Use low-temperature moderator model
        llm, _ = get_llm(mode, "moderator", model_choice)
        
        user_prompt = (
            f"Search Context:\n{search_context}\n\n"
            f"Tutor's Statement to Verify:\n{statement}\n"
        )
        
        messages = [
            ("system", FAITHFULNESS_SYSTEM_PROMPT),
            ("user", user_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        return parse_json_safely(response.content)
    except Exception as e:
        print(f"[WS Warning] Error during faithfulness check: {e}")
        # Default to faithful to prevent blocking the conversation in case of API/rate limit failure
        return {"faithful": True, "reason": ""}

def extract_fallback_tool_calls(response_msg) -> None:
    """
    Defensive parser for local/cloud models that output JSON tool calls directly 
    in the text content instead of populating response_msg.tool_calls.
    Extracts the tool call, builds the tool_calls metadata list, and strips the 
    raw JSON block from the text content. Handles nested JSON objects.
    """
    import uuid
    import json
    import re
    
    content = response_msg.content.strip()
    if not content:
        return
        
    tool_calls = list(response_msg.tool_calls) if response_msg.tool_calls else []
    
    # Locate all matching outer JSON object blocks starting with {"name":
    start_idx = 0
    while True:
        # Search for potential JSON pattern, accepting optional spacing
        match_start = re.search(r'\{\s*"name"\s*:\s*"', content[start_idx:])
        if not match_start:
            break
            
        idx = start_idx + match_start.start()
        
        # Track opening/closing braces to match the outer JSON block
        brace_count = 0
        end_idx = -1
        for i in range(idx, len(content)):
            char = content[i]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
                    
        if end_idx != -1:
            candidate = content[idx:end_idx]
            try:
                parsed = json.loads(candidate)
                tool_name = parsed.get("name")
                if tool_name in ["search_wikipedia", "web_search"]:
                    tool_args = parsed.get("args") or parsed.get("parameters") or parsed.get("arguments") or {}
                    
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except Exception:
                            tool_args = {"query": tool_args}
                            
                    call_id = f"call_{uuid.uuid4().hex[:8]}"
                    tool_call = {
                        "name": tool_name,
                        "args": tool_args,
                        "id": call_id
                    }
                    tool_calls.append(tool_call)
                    
                    # Strip the candidate string and any wrapping code blocks from content
                    content = content.replace(candidate, "").strip()
                    content = re.sub(r'```json\s*```', '', content)
                    content = re.sub(r'```\s*```', '', content)
                    content = content.strip()
                    # Reset search index since we modified the string length
                    start_idx = 0
                    continue
            except Exception as parse_err:
                print(f"[WS Warning] Failed parsing candidate tool call JSON: {parse_err}")
                
        start_idx = idx + 8
        if start_idx >= len(content):
            break
            
    # Update the message object in-place
    response_msg.content = content
    response_msg.tool_calls = tool_calls

async def tutor_node(state: DiscussionState, config=None) -> DiscussionState:
    messages = state["messages"]
    topic = state["topic"]
    mode = state["mode"]
    model_choice = state.get("model", "")
    
    # Retrieve MCP tools and websocket from config parameters
    mcp_tools = []
    websocket = None
    if config and "configurable" in config:
        mcp_tools = config["configurable"].get("mcp_tools", [])
        websocket = config["configurable"].get("websocket", None)
        
    llm, model_name = get_llm(mode, "tutor", model_choice)
    
    # Bind tools if available
    llm_to_use = llm
    using_native_tools = False
    if mcp_tools:
        llm_to_use = llm.bind_tools(mcp_tools)
        using_native_tools = True
    
    history_str = ""
    for msg in messages:
        sender_name = "Tutor" if msg["sender"] == "tutor" else "Learner"
        history_str += f"{sender_name}: {msg['content']}\n"
    
    system_prompt = (
        f"You are a helpful, patient, and knowledgeable AI Tutor teaching a student about '{topic}'.\n"
        "Your goal is to explain concepts clearly, keep explanations brief (under 3 sentences), "
        "and ask simple questions or pose quick quizzes to check the learner's understanding.\n"
        "CRITICAL: You must use the search tools at least once per turn to look up facts, definitions, history, "
        "or summaries related to the topic. This ensures accuracy and verifies the latest context.\n"
        "If you do not have native tool-calling capabilities, you can execute a tool by outputting a JSON block in your text response exactly like this:\n"
        "{\"name\": \"search_wikipedia\", \"parameters\": {\"query\": \"<search query>\"}}\n"
        "or\n"
        "{\"name\": \"web_search\", \"parameters\": {\"query\": \"<search query>\"}}\n"
        "Always stay in character as the Tutor. Do not say things like 'Sure, here is the response:'."
    )
    
    user_prompt = (
        f"Conversation history so far:\n{history_str}\n"
        "Generate the next response as the Tutor. If the history is empty, introduce the topic and ask the learner "
        "what they already know about it or what specific questions they have."
    )
    
    messages_to_send = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        try:
            # Run async invoke (try native tool binding first)
            response = await llm_to_use.ainvoke(messages_to_send)
        except Exception as api_err:
            api_err_str = str(api_err)
            if using_native_tools and ("does not support tools" in api_err_str or "400" in api_err_str or "not supported" in api_err_str):
                print(f"[WS Warning] Local model does not support native tools. Falling back to text-prompted tools. Error: {api_err}")
                llm_to_use = llm # Fallback to unbound LLM
                using_native_tools = False
                response = await llm_to_use.ainvoke(messages_to_send)
            else:
                raise api_err
                
        extract_fallback_tool_calls(response)
        
        # Tool execution ReAct loop (limit to max 3 iterations to avoid loops)
        loop_limit = 3
        loop_count = 0
        
        while response.tool_calls and loop_count < loop_limit:
            loop_count += 1
            messages_to_send.append(response)
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"[WS DEBUG] Tutor agent requests tool: {tool_name} with args: {tool_args}")
                
                # Send WebSocket status update if connected
                if websocket:
                    try:
                        action_name = "Wikipedia" if tool_name == "search_wikipedia" else "the web"
                        query_val = tool_args.get("query", "")
                        await websocket.send_json({
                            "type": "status",
                            "status": f"🔍 Searching {action_name} for '{query_val}'..."
                        })
                    except Exception as ws_err:
                        print(f"[WS Warning] Failed to send tool call status: {ws_err}")
                
                # Find matching tool
                target_tool = next((t for t in mcp_tools if t.name == tool_name), None)
                if target_tool:
                    try:
                        tool_result = await target_tool.ainvoke(tool_args)
                        print(f"[WS DEBUG] Tool '{tool_name}' result fetched successfully.")
                    except Exception as tool_err:
                        tool_result = f"Error executing tool: {tool_err}"
                else:
                    tool_result = f"Tool '{tool_name}' not found."
                
                if using_native_tools:
                    messages_to_send.append(
                        ToolMessage(
                            content=str(tool_result),
                            name=tool_name,
                            tool_call_id=tool_call["id"]
                        )
                    )
                else:
                    # Fallback text-prompted tool result formatting (uses standard user role to avoid serialization errors)
                    messages_to_send.append(
                        HumanMessage(
                            content=f"[System Search Result for '{tool_name}']:\n{tool_result}\n"
                                    f"Please use this search context to answer the student's question. Remember to keep it under 3 sentences."
                        )
                    )
                
            try:
                response = await llm_to_use.ainvoke(messages_to_send)
            except Exception as api_err:
                api_err_str = str(api_err)
                if using_native_tools and ("does not support tools" in api_err_str or "400" in api_err_str or "not supported" in api_err_str):
                    print(f"[WS Warning] Local model does not support native tools during ReAct loop. Falling back to text-prompted tools.")
                    llm_to_use = llm
                    using_native_tools = False
                    response = await llm_to_use.ainvoke(messages_to_send)
                else:
                    raise api_err
                    
            extract_fallback_tool_calls(response)
            
        content = response.content.strip()
        
        # Safe Fallback: If content is empty (e.g., due to loop limit or serialization bugs), synthesize a response
        if not content:
            search_texts = []
            for msg in messages_to_send:
                if isinstance(msg, ToolMessage) or (isinstance(msg, HumanMessage) and "[System Search Result" in msg.content):
                    search_texts.append(msg.content)
            
            if search_texts:
                search_context = "\n\n---\n\n".join(search_texts)
                print("[WS Warning] Tutor final response text was empty but search results were retrieved. Synthesizing safe summary...")
                try:
                    search_summary_prompt = (
                        "Generate a friendly, brief (under 3 sentences) educational response to the student "
                        "summarizing the main points of these search results:\n"
                        f"{search_context}\n"
                        "Do not output any JSON or tool calls."
                    )
                    summary_msg = await llm.ainvoke([
                        SystemMessage(content="You are a helpful AI Tutor."),
                        HumanMessage(content=search_summary_prompt)
                    ])
                    content = summary_msg.content.strip()
                except Exception as e:
                    print(f"[WS Warning] Failed to synthesize summary: {e}")
            
            if not content:
                content = f"I've fetched some details about '{topic}'. What specific questions do you have about it?"
        
        # Self-correction / Faithfulness validation loop
        search_texts = []
        for msg in messages_to_send:
            if isinstance(msg, ToolMessage):
                search_texts.append(f"Source Tool ({msg.name}):\n{msg.content}")
                
        if search_texts and not content.startswith("[Tutor Error:"):
            search_context = "\n\n---\n\n".join(search_texts)
            print(f"[WS DEBUG] Checking Tutor output faithfulness against {len(search_texts)} search tool results...")
            verification = await check_tutor_faithfulness(content, search_context, mode, model_choice)
            
            if not verification.get("faithful", True):
                reason = verification.get("reason", "Claims not fully supported by search context.")
                print(f"[WS Warning] Tutor output flagged as unfaithful: {reason}")
                
                if websocket:
                    try:
                        await websocket.send_json({
                            "type": "status",
                            "status": "⚠️ Recalibrating facts and resolving contradictions..."
                        })
                    except Exception as ws_err:
                        print(f"[WS Warning] Failed to send status update: {ws_err}")
                
                # Append correction prompt and run a one-time regeneration
                # We strictly forbid tool calls or JSON blocks during this correction step to avoid nested loops
                correction_prompt = (
                    f"CRITICAL GROUNDEDNESS CHECK WARNING: Your proposed response contained statements not supported "
                    f"by the retrieved search results. Reason for failure: {reason}\n"
                    "You must regenerate your response. Keep it under 3 sentences and ground it STRICTLY "
                    "in the search results provided. If a fact is not explicitly supported, omit it or say you do not know.\n"
                    "CRITICAL: Do NOT execute any search tools or output any JSON blocks. Output ONLY plain text for the student."
                )
                
                messages_to_send.append(response)
                messages_to_send.append(HumanMessage(content=correction_prompt))
                
                try:
                    corrected_response = await llm_to_use.ainvoke(messages_to_send)
                    extract_fallback_tool_calls(corrected_response)
                    regen_content = corrected_response.content.strip()
                    if regen_content:
                        content = regen_content
                        print("[WS DEBUG] Tutor output successfully regenerated and corrected.")
                    else:
                        print("[WS Warning] Tutor regenerated response was empty. Retaining original content with disclaimer.")
                        content = "[Fact Correction Alert] Omitted unverified details. " + content
                except Exception as regen_err:
                    print(f"[WS Warning] Failed to regenerate Tutor response: {regen_err}")
                    
    except Exception as e:
        content = f"[Tutor Error: Could not generate response. Details: {str(e)}]"
        model_name = "Error Node"
        
    new_message: ChatMessage = {
        "sender": "tutor",
        "content": content,
        "model_used": model_name
    }
    
    return {
        **state,
        "messages": messages + [new_message],
        "turn_count": state["turn_count"] + 1
    }

def learner_node(state: DiscussionState) -> DiscussionState:
    messages = state["messages"]
    topic = state["topic"]
    mode = state["mode"]
    model_choice = state.get("model", "")
    
    llm, model_name = get_llm(mode, "learner", model_choice)
    
    history_str = ""
    for msg in messages:
        sender_name = "Tutor" if msg["sender"] == "tutor" else "Learner"
        history_str += f"{sender_name}: {msg['content']}\n"
        
    system_prompt = (
        f"You are a curious and polite AI Learner who is actively learning about '{topic}' from a Tutor.\n"
        "Your goal is to ask insightful questions, try your best to answer the tutor's quizzes/questions, "
        "and show what you understand. Keep your responses short (under 3 sentences).\n"
        "Always stay in character as the Learner. Do not say things like 'Sure, here is the response:'."
    )
    
    user_prompt = (
        f"Conversation history so far:\n{history_str}\n"
        "Generate the next response as the Learner."
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        content = response.content.strip()
    except Exception as e:
        content = f"[Learner Error: Could not generate response. Details: {str(e)}]"
        model_name = "Error Node"
        
    new_message: ChatMessage = {
        "sender": "learner",
        "content": content,
        "model_used": model_name
    }
    
    return {
        **state,
        "messages": messages + [new_message],
        "turn_count": state["turn_count"] + 1
    }

def evaluator_node(state: DiscussionState) -> DiscussionState:
    messages = state["messages"]
    topic = state["topic"]
    mode = state["mode"]
    model_choice = state.get("model", "")
    
    llm, model_name = get_llm(mode, "evaluator", model_choice)
    
    history_str = ""
    for msg in messages:
        sender_name = "Tutor" if msg["sender"] == "tutor" else "Learner"
        history_str += f"{sender_name}: {msg['content']}\n"
        
    system_prompt = (
        "You are an objective AI Tutor Evaluator. Your job is to analyze the debate/conversation between the Tutor and the Learner "
        f"about the topic '{topic}' and grade the Learner's performance and conceptual understanding.\n"
        "You MUST respond ONLY with a raw JSON object (no markdown block, no ```json, no extra text). "
        "The JSON object must follow this schema exactly:\n"
        "{\n"
        "  \"score\": <integer between 1 and 10>,\n"
        "  \"summary\": \"<2-3 sentence overall critique of the learner's understanding>\",\n"
        "  \"strengths\": [\"<strength 1>\", \"<strength 2>\"],\n"
        "  \"gaps\": [\"<concept misunderstood or missed 1>\", \"<concept misunderstood 2>\"],\n"
        "  \"recommendations\": [\"<actionable reading or advice 1>\", \"<actionable reading or advice 2>\"]\n"
        "}"
    )
    
    user_prompt = (
        f"Here is the dialogue transcript:\n{history_str}\n"
        "Please evaluate the Learner and return the raw JSON scorecard."
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        content = response.content.strip()
        
        # Clean potential markdown wrapping
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
            
        evaluation_data = json.loads(content)
    except Exception as e:
        evaluation_data = {
            "score": 0,
            "summary": f"Could not perform evaluation. Error: {str(e)}",
            "strengths": ["Evaluation failed"],
            "gaps": ["Error parsing response"],
            "recommendations": ["Ensure API keys are configured and local models are pulled"]
        }
        
    return {
        **state,
        "evaluation": evaluation_data
    }
