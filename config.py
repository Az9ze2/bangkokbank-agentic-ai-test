"""Loads configuration and builds the shared Azure OpenAI chat client."""

import os

from langchain_openai import AzureChatOpenAI


def get_llm() -> AzureChatOpenAI:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not endpoint or not api_key:
        raise RuntimeError(
            "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY. "
            "Copy .env.example to .env and fill in the values "
            "(request the subscription key from kanit.mekritthikrai@bangkokbank.com)."
        )

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        azure_deployment=deployment,
        api_version=api_version,
    )
