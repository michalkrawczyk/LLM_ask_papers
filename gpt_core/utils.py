import openai

from itertools import islice
from time import sleep
import logging


def _prompt_count(attempts: int = 1):
    # Wrapper to create used prompt counter
    _prompt_count.counter += attempts


_prompt_count.counter = 0


def check_prompt_rate(limit_rate: int = 3):
    """ Check number of used prompts for ensuring not
    exceeding limit rate of GPT (3 prompts / min)

    If number is exceeded - sleep for minute to norm prompt rate

    limit_rate: int
       maximum rate of prompts per number for model.
       Default is 3 prompts / minute for chatGPT

    """
    if _prompt_count.counter > 0 and \
            _prompt_count.counter % (limit_rate + 1) == 0:
        sleep(60)

    return _prompt_count.counter


def create_batches(iterable, batch_size: int):
    """ Split iterable into smaller batches"""
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def get_completion(prompt: str, model="gpt-3.5-turbo", max_attempts: int = 3):
    """ Get Completion from OpenAI model for given prompt

    Parameters
    ----------
    prompt: str
        Prompt (Instruction) for model
    model:
        OpenAI model to use
    max_attempts: int
        Maximum number of failed attempts before raising error

    Returns
    -------
    String with response for prompt

    """
    messages = [{"role": "user", "content": prompt}]

    for attempts in range(max_attempts):
        try:
            # Count and check current number of used responses ratio per minute
            _prompt_count()
            check_prompt_rate()

            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,  # this is the degree of randomness of the model's output
            )
            break   # Everything ok, no need to attempt further
        except Exception as err:
            logging.warning(f"Failed to obtain response "
                            f"(Attempt {attempts + 1}/{max_attempts} \n"
                            f"error: {err}")

            if attempts == (max_attempts - 1):
                raise err

            # Sleep 3 seconds after failed attempt, due to most probably busy server
            sleep(3)

    return response.choices[0].message["content"]
