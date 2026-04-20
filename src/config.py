import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(override=True)


def get_llm(model: str = "gpt-4o", temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.environ["OPENAI_API_KEY"],
    )


def get_embeddings(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model, api_key=os.environ["OPENAI_API_KEY"])
