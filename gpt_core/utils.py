import openai

from itertools import islice


def create_batches(iterable, batch_size: int):
    """ Split iterable into smaller batches"""
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def get_completion(prompt: str, model="gpt-3.5-turbo"):
    """ Get Completion from OpenAI model for given prompt

    Parameters
    ----------
    prompt: str
        Prompt (Instruction) for model
    model:
        OpenAI model to use

    Returns
    -------
    String with response for prompt

    """
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]
