from langchain_core.prompts import PromptTemplate

CHAT_PROMPT = PromptTemplate(
    template="""Answer using only the provided context briefly and pointwise.
    Context:
    {context}

    Question:
    {question}
    """,

    input_variables=["context", "question"]
)
