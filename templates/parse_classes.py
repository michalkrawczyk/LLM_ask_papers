from pydantic import BaseModel, Field
from typing import List, Union

# Note: description in pydentic classes is included in output parser


class ShortInfoSummary(BaseModel):
    model_name: str = Field(
        description="Name of the model if provided"
    )
    model_category: str = Field(
        description="Model Category (e.g Object Detection, NLP or image generation)",

    )
    sota: int = Field(description="Boolean - Is this model State-of-the-Art?")
    new_features: Union[List[str], str] = Field(
        description="Introduced new model components, layers or other features, as keywords, each seperated by commas",
    )
    new_strategies: Union[List[str], str]  = Field(
        description="New strategies introduced, as keywords"
    )
    date: str = Field(description="Date of the paper")
