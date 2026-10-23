
from app.ingestion import ingest_documents
from app.rag import ask_question


def main():
    print("Starting RAG application...\n")

    # Prepare the knowledge base once
    ingest_documents()

    print("\nKnowledge assistant is ready!")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = ask_question(question)

            print("\nAssistant:")
            print(result["answer"])

            print("\nSources:")
            for index, source in enumerate(
                result["sources"], start=1
            ):
                print(
                    f"\n{index}. {source['source']} "
                    f"(score: {source['score']})"
                )
                print(source["text"])

            print("\n" + "-" * 60 + "\n")

        except Exception as exc:
            print(f"\nRequest failed: {exc}\n")


if __name__ == "__main__":
    main()
