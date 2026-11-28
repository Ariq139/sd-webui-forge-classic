"""Native API contracts; model paths and execution settings come from the active preset."""
from typing import Optional
from pydantic import BaseModel, Field


class NativeRequest(BaseModel):
    prompt: str = Field(min_length=1)
    width: Optional[int] = Field(default=None, ge=32, le=4096)
    height: Optional[int] = Field(default=None, ge=32, le=4096)
    steps: Optional[int] = Field(default=None, ge=1, le=150)
    guidance: Optional[float] = Field(default=None, ge=0, le=24)
    seed: int = Field(default=-1, ge=-1, lt=2**63)


class IdeogramRequest(NativeRequest):
    num_images: int = Field(default=1, ge=1, le=8)


class LTXRequest(NativeRequest):
    negative_prompt: str = ""
    frames: Optional[int] = Field(default=None, ge=9, le=241)
    image: Optional[str] = Field(default=None, max_length=24000000)
