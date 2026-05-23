from typing import TypedDict, List, Literal

class ChatMessage(TypedDict):
    sender: Literal["tutor", "learner"]
    content: str
    model_used: str

class DiscussionState(TypedDict):
    topic: str
    mode: Literal["local", "cloud"]
    model: str  # The specific model chosen by the user
    messages: List[ChatMessage]
    turn_count: int
    evaluation: dict
