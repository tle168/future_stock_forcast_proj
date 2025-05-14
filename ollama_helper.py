import requests
from openai import OpenAI

AI_PROMPT = """
                You are a Stock Trader specializing in Technical Analysis at a top financial institution.
                Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
                Base your recommendation only on the candlestick chart and the displayed technical indicators.
                First, provide the recommendation, then, provide your detailed reasoning.
                Always respond in Vietnamese.
            """
messages=[
            {"role": "user", "content": AI_PROMPT}
        ]
def ask_ollama(prompt, model="gemma3"):
    # This is an example of how to use the OpenAI Python client with the Ollama API.
    client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
    )
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        stream=False,
        messages=messages
    )
    reply=response.choices[0].message.content

    #if response.status_code == 200:
    return reply
    #
