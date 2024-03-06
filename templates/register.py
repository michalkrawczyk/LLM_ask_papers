"""Those register functions are provided to quickly register prompts to given prompt holder.
They're outside PromptHolder class,
as they are also made as convenient shortcuts to add prompts to default prompt holder.
"""
from typing import List, Optional

from langchain.schema import BasePromptTemplate, BaseOutputParser
from langchain.prompts import PromptTemplate

from .prompt_holders import PromptHolder

DEFAULT_PROMPT_REGISTER = PromptHolder()


def register_prompt(
    name: str,
    prompt: BasePromptTemplate,
    prompt_register: PromptHolder = DEFAULT_PROMPT_REGISTER,
    force_reload: bool = False,
):
    # Note:
    prompt_register.load_defined_prompt(
        name=name, prompt=prompt, force_reload=force_reload
    )


def create_and_register_prompt(
    name: str,
    template: str,
    input_variables: List[str],
    prompt_register: PromptHolder = DEFAULT_PROMPT_REGISTER,
    force_reload: bool = False,
    output_parser: Optional[BaseOutputParser] = None,
):
    if not force_reload and name in prompt_register:
        # Safer here to raise an error than use possibly wrong prompt
        raise ValueError(f"Prompt {name} already defined")

    prompt = PromptTemplate(
        template=template, input_variables=input_variables, output_parser=output_parser
    )
    prompt_register.load_defined_prompt(name=name, prompt=prompt)


# TODO: register prompt from file
