from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="./db",
    embedding_function=embeddings
)

retriever = db.as_retriever()

llm = ChatOllama(
    model="llama3.2"
)

while True:

    question = input("\nAsk: ")

    if question == "exit":
        break

    docs = retriever.invoke(question)

    context = "\n".join(
        [d.page_content for d in docs]
    )

    prompt = f"""
Use only this context.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)