import openai
import yaml

from .prompts import get_description_json, get_summary_3

import os

if os.path.isfile('openai_key.yaml'):
    with open('openai_key.yaml', 'r') as f:
        # Read API KEY for ChatGPT
        openai.api_key = yaml.safe_load(f)["openai_api_key"]

else:
    openai.api_key = os.environ.get("OPENAI_API_KEY")


with open('prompts.yaml', 'r') as f:
    # Read prompts that will be used to make summaries
  PROMPTS = yaml.safe_load(f)

__all__ = [
    "get_description_json",
    "get_summary_3",
    "PROMPTS"
]