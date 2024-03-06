"""
PromptHolder class is used to store or create predefined langchain prompts for later usage.
It is created to provide invidual set of prompts for different PaperDatasetLC instances
(among others for different models or purposes - e.g. for one dataset with medical topics and one for financial).

If user don't want to use multiple instances,
he can use default prompt holder (DEFAULT_PROMPT_REGISTER) from 'templates' module,
which is used by default when no PromptHolder is provided.
"""

import logging
import os
from typing import Dict
from dataclasses import dataclass, field

from langchain.prompts import load_prompt
from langchain.schema import BasePromptTemplate

# ROOT_PATH = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


@dataclass
class PromptHolder:
    PROMPTS: Dict[str, BasePromptTemplate] = field(default_factory=dict)
    # TODO: Think about adding test for prompt content checks

    def load_prompt_file(self, name: str, prompt_path: str):
        try:
            if os.path.splitext(prompt_path)[-1] in [".yaml", ".py", ".json"]:
                raise ValueError("Invalid prompt file type")

            if name in self.PROMPTS:
                raise ValueError(f"Prompt {name} already defined")

            prompt = load_prompt(prompt_path)
            self.PROMPTS[name] = prompt

        except Exception as err:
            logger.error(f"Error loading prompt {name} from {prompt_path}: {err}")

    def load_defined_prompt(
        self, name: str, prompt: BasePromptTemplate, force_reload: bool = False
    ):
        if name not in self.PROMPTS or force_reload:
            self.PROMPTS[name] = prompt
        else:
            logger.warning(f"Prompt {name} already defined, skipping.")

    def delete_prompt(self, name: str):
        self.PROMPTS.pop(name, None)

    @property
    def get_prompts(self) -> Dict[str, BasePromptTemplate]:
        return self.PROMPTS

    def __getitem__(self, name: str) -> BasePromptTemplate:
        return self.PROMPTS[name]

    def __contains__(self, item: str) -> bool:
        return item in self.PROMPTS.keys()

    def get(self, name: str) -> BasePromptTemplate:
        return self.PROMPTS.get(name, None)
