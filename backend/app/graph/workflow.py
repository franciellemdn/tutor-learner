from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.graph.state import DiscussionState
from app.graph.nodes import tutor_node, learner_node, evaluator_node

# Router logic
def router_condition(state: DiscussionState) -> str:
    # Stop debate and send to evaluator after 6 turns
    if state["turn_count"] >= 6:
        return "evaluator"
    
    last_sender = state["messages"][-1]["sender"]
    if last_sender == "tutor":
        return "learner"
    else:
        return "tutor"

# Compile LangGraph Workflow
workflow = StateGraph(DiscussionState)

# Add nodes
workflow.add_node("tutor", tutor_node)
workflow.add_node("learner", learner_node)
workflow.add_node("evaluator", evaluator_node)

# Set entry point
workflow.set_entry_point("tutor")

# Set conditional edges
workflow.add_conditional_edges(
    "tutor",
    router_condition,
    {
        "learner": "learner",
        "evaluator": "evaluator"
    }
)
workflow.add_conditional_edges(
    "learner",
    router_condition,
    {
        "tutor": "tutor",
        "evaluator": "evaluator"
    }
)
workflow.add_edge("evaluator", END)

# Export compiled workflow singleton
checkpointer = MemorySaver()
discussion_graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["learner"])
