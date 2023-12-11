from pydantic import BaseModel, Field
from typing import List


class ShortInfoSummary(BaseModel):
    """Short summary of the research paper, focused on new features/strategies introduced in the paper"""
    model_name: str = Field(
        description="Name of the model if provided", default="Unknown"
    )
    model_category: str = Field(
        description="Model Category (e.g Object Detection, NLP or image generation)",
        default="Unknown",
    )
    sota: bool = Field(description="Is this model State-of-the-Art?", default=False)
    new_features: List[str] = Field(
        description="Introduced model components, layers or other features, as keywords",
        default=[],
    )
    new_strategies: List[str] = Field(
        description="New strategies introduced, as keywords", default=[]
    )
    date: str = Field(description="Date of the paper", default="Unknown")
