# ? so now we are creating a graph
# ? and then first things you create is a state

from langgraph.graph import MessagesState
from dataclasses import dataclass, field
from pydantic import BaseModel, field_validator
import os

# 1) typed DICT

from typing import TypedDict


class State(TypedDict):
    topic: str
    summary: str
    score: str

# 2) pydantic approach
# it is good at data validation and type checking at
# runtime


class State(BaseModel):
    topic: str
    score: int
    summary: str = ""

    @field_validator
    def score_positive(cls, v):
        if v < 0:
            raise ValueError("score must be positive")


# python data classes

# standard python dataclass but it is used very rerelty


@dataclass
class State:
    topic: str = ""
    score: int = ""
    message: list = field(default_factory=list)


class State(MessagesState):
    # message field is already included with add_message reducer
    # just add your extra fields
    name: str = ""
    language: str = ""
