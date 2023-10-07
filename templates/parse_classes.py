from pydantic import BaseModel, Field
from typing import List

# TODO: Think about dynamic created BaseModels by yaml file


class ShortInfoSummary(BaseModel):
    model_name: str = Field(description="Name of the model if provided", default="Unknown")
    model_category: str = Field(description="Model Category (e.g Object Detection, NLP or image generation)", default="Unknown")
    sota: bool = Field(description="Is this model State-of-the-Art?")
    new_features: List[str] = Field(description="New features introduced, as keywords")
    new_strategies: List[str] = Field(description="New strategies introduced, as keywords")
    date: str = Field(description="Date of the paper", default="Unknown")

