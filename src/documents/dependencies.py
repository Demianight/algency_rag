from llama_cloud_services import LlamaParse

from src.settings import settings


def get_llama_parser():
    return LlamaParse(
        api_key=settings.llama.api_key,
        verbose=True,
        language="ru",
    )
