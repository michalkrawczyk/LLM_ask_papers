from pydantic import BaseModel, Field
from typing import List

# Note: description in pydentic classes is included in output parser


class ShortInfoSummary(BaseModel):
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
        description="New strategies introduced, as keywords, separated by commas", default=[]
    )
    date: str = Field(description="Date of the paper", default="Unknown")
