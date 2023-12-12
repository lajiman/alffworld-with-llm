import os
import time

import openai
from openai import OpenAI


def llm(llm_messages, llm_model = "gpt-3.5-turbo-16k",stop=None) -> str:
    client = OpenAI()
    completion = client.chat.completions.create(
        model=llm_model,
        messages=llm_messages,
        temperature=0.0,
        max_tokens=100,
        frequency_penalty=0.5,
        presence_penalty=0.0,
        stop=stop,
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    client = OpenAI()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Hello, Nice to meet you"}
    ]
    )

    print(completion.choices[0].message.content)