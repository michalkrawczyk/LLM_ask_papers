from tqdm import tqdm

from arxiv_utils import PaperData
from .utils import get_completion, create_batches
from gpt_core import PROMPTS

import json
from typing import Union


def get_description_json(text: Union[str, PaperData], max_tokens: int = 1100) -> dict:
    """ Get short description of paper in JSON format

    Parameters
    ----------
    text: str
        Text of the page
        .. warning:: If text contain more than 1100 words will be trimmed to that count

    max_tokens: int
        Maximum number of tokens(words) from paper used in prompt

    Returns
    -------

    JSON (as dict) with short description of paper

    """
    # fit text into max token ammount
    if isinstance(text, str):
        # Trim text to fit prompt
        text_to_describe = [' '.join(word) for word in text.split(' ')[:max_tokens]]
    else:
        # Attempt to concatenate pages with maximum length
        # to fit more of them into prompt
        text_to_describe = text.join_pages_by_length(max_tokens)[0].text

    prompt = f"""{PROMPTS["short_decription_json"]} '''{text_to_describe}'''"""

    # prompt = f"""
    # Identify the following items from given text, delimited by triple backticks:
    # - Model Name
    # - Model category(e.g Object Detection, NLP or image generation)
    # - SOTA: if Model is State-of-the-Art
    # - New Features: new features introduced
    # - Year: Year of publish

    # Format your response as a JSON object with \
    # "Model Name", "Model Category", "SOTA", "New Features" and "Year" as the keys.
    # If the information isn't present, use "unknown" \
    # as the value.
    # Make your response as short as possible.
    # Format the SOTA value as a boolean.

    # Review text: '''{text_to_describe}'''
    # """

    response = get_completion(prompt)

    return json.loads(response)


def get_summary(text: Union[str, PaperData],
                max_tokens_per_prompt: int = 1200) -> str:
    """ Obtain Summary for research paper

    Parameters
    ----------
    text: Union[str, PaperData]
        String or PaperData object containing text to summarize
    max_tokens_per_prompt: int
        Limit of words feeded to prompt, to not exceed model's number of maximum tokens.

    Returns
    -------
    response: str
        Response from model, containing created summary


    """
    text_batches = []

    if isinstance(text, str):
        # Create batches from text, if it's too long
        text_batches = [' '.join(word) for word in \
                        create_batches(text.split(' '), max_tokens_per_prompt)]
    else:
        # Make batches from pages by joining limited with maximum number of words
        text_batches = [t.text for t in
                        text.join_pages_by_length(max_tokens_per_prompt)]


    # You have to identify the following items from given text, delimited by triple backticks:

    # - New Features: (listed each of names of new features, functions and functionalities)
    # - New Stategies: (listed new stategies and techniques)
    # - Problems: (listed tackled problems and approaches)
    # - Design: (network design)
    # - maximum of three sentences for obtained results.

    # Format your response to pointed answears for each category.
    # For each feature in identified new features write on the end few sentences of summary

    # With each next text you have to fill your previous response with missing informations"""

    base_prompt = f"""{PROMPTS["summary"]} ''''{text_batches[0]}'''"""

    responses = [get_completion(base_prompt)]
    # Counter for ensuring not exceeding limit rate of GPT (3 prompts / min)

    for text_batch in tqdm(text_batches[1:], desc="Scanning document"):

        continue_prompt = f"""{PROMPTS["continue_summary"]}
          text: '''{text_batch}'''
    
          summary to fill: {responses[-1]}
    
          """
        # "You have to identify the following items from given text, delimited by triple backticks: \n
        # - New Features: (listed every name for new features, functions and functionalities
        #   and corresponding components)
        # - New Stategies: (listed new stategies and techniques)
        # - Problems: (listed tackled problems and approaches)
        # - Design: (network design) \n
        #
        #    After that fill the summary in quotes with missing informations."

        responses.append(get_completion(continue_prompt))

    return responses
