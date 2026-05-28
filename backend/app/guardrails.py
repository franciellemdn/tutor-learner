import json
import re
from app.models import get_llm

GUARDRAIL_SYSTEM_PROMPT = """You are a content safety moderator for an educational AI agent debate system.
Your job is to analyze the user's proposed debate topic and determine if it is safe and appropriate.

Strictly block any topics that contain or promote:
1. Self-harm, suicide, or self-injury.
2. Hate speech, discrimination, harassment, or slurs.
3. Severe violence, weapons, explosives, murder, or illegal acts.
4. Sexual content, pornography, or explicit adult themes.
5. Cyberattacks, malware, hacking instructions, or system exploits.
6. Highly harmful, toxic, or dangerous activities.

Guidelines:
- Sensitive academic, historical, political, or philosophical topics (e.g., "Ethics of capital punishment", "Why do wars happen", "French Revolution violence", "History of fascism") must be marked as SAFE as long as they do not promote, encourage, incite, or provide instructions for harm.
- If the topic is safe, set "safe" to true and "reason" to an empty string "".
- If the topic is unsafe, set "safe" to false and "reason" to a clear, user-friendly explanation of why it was blocked.

You must respond in strict JSON format with exactly these two keys:
{
  "safe": boolean,
  "reason": string
}
Do not return any other text besides the JSON object."""

def parse_json_safely(text: str) -> dict:
    text = text.strip()
    
    # Strip markdown code blocks if present
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
            
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Robust manual parsing fallback in case of malformed model outputs
        lower_text = text.lower()
        safe_val = True
        if '"safe": false' in lower_text or '"safe":false' in lower_text:
            safe_val = False
            
        reason_match = re.search(r'"reason"\s*:\s*"(.*?)"', text)
        reason_val = reason_match.group(1) if reason_match else ""
        if not safe_val and not reason_val:
            reason_val = "Topic was flagged as potentially harmful or violating content safety guidelines."
            
        return {"safe": safe_val, "reason": reason_val}

async def check_topic_safety(topic: str, mode: str, model_name: str) -> dict:
    """
    Analyzes the safety of the proposed topic.
    Returns:
        dict: {"safe": bool, "reason": str}
    """
    try:
        llm, _ = get_llm(mode, "moderator", model_name)
        
        messages = [
            ("system", GUARDRAIL_SYSTEM_PROMPT),
            ("user", f"Analyze this topic: \"{topic}\"")
        ]
        
        # Invoke LLM asynchronously
        response = await llm.ainvoke(messages)
        response_text = response.content
        return parse_json_safely(response_text)
    except Exception as e:
        print(f"[Guardrails Warning] Error during safety check: {e}")
        # Default to safe in case of API outages to prevent blocking the service completely
        return {"safe": True, "reason": ""}
