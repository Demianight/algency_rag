from langchain.chains import RetrievalQA


def ask_gpt(question: str, llm, vectorstore, top_k: int = 5) -> dict:
    """
    Ask a question to the LLM and return the answer along with source documents.
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
        return_source_documents=True,
    ).invoke({"query": question})
