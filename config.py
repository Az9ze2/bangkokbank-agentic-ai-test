"""Loads configuration and builds the shared LLM chat client.

Supports either a standard OpenAI API key (OPENAI_API_KEY) or Bangkok Bank's
Azure OpenAI deployment (AZURE_OPENAI_*). Standard OpenAI is preferred if both
are present.
"""

import os

from langchain_openai import AzureChatOpenAI, ChatOpenAI


def get_llm():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        return ChatOpenAI(api_key=openai_api_key, model=model)

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if endpoint and azure_api_key:
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=azure_api_key,
            azure_deployment=deployment,
            api_version=api_version,
        )

    raise RuntimeError(
        "Missing LLM credentials. Copy .env.example to .env and set either "
        "OPENAI_API_KEY (standard OpenAI) or AZURE_OPENAI_ENDPOINT + "
        "AZURE_OPENAI_API_KEY (Bangkok Bank's Azure deployment, request the "
        "key from kanit.mekritthikrai@bangkokbank.com)."
    )
