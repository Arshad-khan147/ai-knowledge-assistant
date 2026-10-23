
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import OPENROUTER_API_KEY, LLM_MODEL
from app.retrieval import retrieve_documents


prompt = ChatPromptTemplate.from_template("""
You are a helpful knowledge assistant.

Answer the question using only the provided context.
If the context does not contain the answer, say so.
Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
""")

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

rag_chain = prompt | llm



def ask_question(question: str, k: int = 3):
    # Initial threshold for cosine similarity.
    # Tune this using tests on your own documents.
    relevance_threshold = 0.70

    results = retrieve_documents(question, k=k)

    # Keep only chunks whose similarity score meets the threshold.
    relevant_results = [
        (document, score)
        for document, score in results
        if float(score) >= relevance_threshold
    ]

    # Do not call the LLM if no retrieved chunk is relevant enough.
    if not relevant_results:
        return {
            "answer": (
                "I couldn't find sufficiently relevant information "
                "in the available documents to answer that question."
            ),
            "sources": [],
        }

    context = "\n\n".join(
        document.page_content
        for document, score in relevant_results
    )

    response = rag_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    sources = [
        {
            "text": document.page_content,
            "source": document.metadata.get("source", "unknown"),
            "page": document.metadata.get("page", "unknown"),
            "score": round(float(score), 4),
        }
        for document, score in relevant_results
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }

