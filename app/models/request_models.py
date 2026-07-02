from typing import List, Optional
from pydantic import BaseModel, Field, validator


class ChatMessage(BaseModel):
    """
    Represents a single message in the conversation history.
    Matches the evaluator's expected format exactly.
    """
    role: str = Field(..., description="The role: 'user' or 'assistant'")
    content: str = Field(..., description="The text content of the message")


class ChatRequest(BaseModel):
    """
    Request schema for POST /chat — stateless, full conversation history.
    
    IMPORTANT: This matches the evaluator's specification exactly:
      { "messages": [{"role": "user", "content": "..."}, ...] }
    
    The last message with role='user' is treated as the latest input.
    """
    messages: List[ChatMessage] = Field(
        ...,
        description="Full conversation history including the latest user message",
        min_length=1
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
                    {"role": "assistant", "content": "Sure. What is seniority level?"},
                    {"role": "user", "content": "Mid-level, around 4 years"}
                ]
            }
        }
    }
