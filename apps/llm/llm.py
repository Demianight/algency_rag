from langchain_openai import ChatOpenAI

from settings import settings

llm = ChatOpenAI(model=settings.openai.qa_model, api_key=settings.openai.api_key)  # type: ignore[arg-type]
