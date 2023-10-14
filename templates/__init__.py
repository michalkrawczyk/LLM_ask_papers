from .parse_classes import ShortInfoSummary
from .prompt_holders import PromptHolder

DEFAULT_PROMPT_REGISTER = PromptHolder()

__all__ = ["DEFAULT_PROMPT_REGISTER",
           "ShortInfoSummary",
           "PromptHolder"]