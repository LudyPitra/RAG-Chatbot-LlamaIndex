import asyncio
from src.rag_service import agent, ctx, cleanup_vectordb


def show_banner():
    print("-" * 30)
    print("🤖 RAG Chatbot - LlamaIndex")
    print("-" * 30)
    print("\nAvailable commands")
    print(" - 'Load <path>' - Load a document")
    print(" - 'exit' - Exit and clean vectordb")
    print("\n💡 Tip: Load a document before making a query")


async def main():
    show_banner()

    try:
        while True:
            print("\n" + "-" * 60)

            prompt = input("Me: ").strip()

            if not prompt:
                continue

            if prompt.lower() == "exit":
                break

            if prompt.lower().startswith("load"):
                file_path = prompt[5:].strip()
                response = await agent.run(
                    f"Use the load_documents function to load the document: {file_path}",
                    ctx=ctx,
                )
            else:
                response = await agent.run(prompt, ctx=ctx)

            print(f"\nAssistant: {response}")

    except KeyboardInterrupt:
        print("\n\n⚠️Interrupted by user")
    finally:
        print("🧹 Cleaning vectordb")
        cleanup_vectordb()
        print("✅ Successfully cleaned!")
        print("\n 👋 See you later")


if __name__ == "__main__":
    asyncio.run(main())
