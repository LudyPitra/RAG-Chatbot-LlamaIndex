import os
import asyncio
import logging
import chromadb
from datetime import datetime
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.docling import DoclingReader

logging.basicConfig(
    level=logging.ERROR,
)

for logger_name in [
    "chromadb",
    "httpx",
    "ollama",
    "llama_index",
    "docling",
    "docling_core",
    "rapidocr",
    "RapidOCR",
]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    logger.propagate = False

Settings.llm = Ollama(
    model="ministral-3:14b", request_timeout=360.0, context_window=8000
)
Settings.embed_model = OllamaEmbedding(
    model_name="mxbai-embed-large",
)
Settings.text_splitter = MarkdownNodeParser()

client = chromadb.PersistentClient(path="./vectordb/")

session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
collection = client.get_or_create_collection(name=session_id)

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

reader = DoclingReader()
current_index = None
current_retriever = None
current_document = None


async def load_documents(ctx: Context, file_path: str) -> str:
    """
    Upload a document (PDF, DOCX, Markdown, CSV) so the chatbot can answer questions about it.

    Arguments: `file_path`: Full or relative path to the file (e.g., `./data/my-file.pdf`)

    Returns:
    Success or error message
    """
    global current_index, current_document, current_retriever

    if not os.path.exists(file_path):
        return f"❌ Error: File not fount in '{file_path}'"

    supported_extensions = [".pdf", ".docx", ".md", ".csv", ".txt"]
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext not in supported_extensions:
        return f"❌ Error: Format '{file_ext}' not supported. Use: {', '.join(supported_extensions)}"

    try:
        print(f"\nLoading documents: {file_path}\n")
        print("⏳Processing and creating embeddings...(may take a few seconds)\n")

        documents = reader.load_data(file_path)

        current_index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
        )
        current_retriever = VectorIndexRetriever(
            index=current_index,
            similarity_top_k=5,
        )

        current_document = os.path.basename(file_path)

        return f"✅ Document '{current_document}' loaded successfully! You can ask your questions."

    except Exception as e:
        return f"❌ Error loading document: {str(e)}"


async def rag_retrieve(ctx: Context, query: str):
    """
    Search relevant sections of a document

    args:
        query: Question or search term.

    retruns:
        Relevant sections of the document or a error message
    """

    global current_retriever, current_document

    if current_retriever is None:
        return "❌No documents loaded. Please use the load_document() function first."

    try:
        nodes = current_retriever.retrieve(query)

        if not nodes:
            return (
                "I didn't find any relevant sections for your question in the document."
            )

        parts = []

        for i, n in enumerate(nodes, start=1):
            node = getattr(n, "node", n)

            if hasattr(node, "get_content"):
                text = node.get_content()
            elif hasattr(node, "text"):
                text = node.text
            else:
                text = str(node)

            parts.append(f"[Node {i}] \n{text}")

        return "\n\n".join(parts)

    except Exception as e:
        return f"❌Search error: {str(e)}"


def cleanup_vectordb():
    "Clears the current session's collection upon closing."

    try:
        print("\n🧹 Cleaning vectordb...")
        client.delete_collection(name=session_id)
        print("✅ Vectordb cleaned successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not clear vectordb: {e}")


agent = FunctionAgent(
    tools=[load_documents, rag_retrieve],
    system_prompt="""
    You are a helpful assistant that helps users analyze documents.
        
    WORKFLOW:
        
    1. If the user asks to load/read/open a document:
        - Use load_document(file_path) with the provided path
        - Confirm loading before answering questions
        
    2. If the user asks questions about documents:
        - ALWAYS use rag_retrieve(query) first to search for information
        - Respond ONLY based on the returned snippets
        - DO NOT make up information that is not in the snippets
        - If no relevant information is found, ask for more context
        
    3. For general questions (not related to documents):
        - Respond normally without using the tools
        
    IMPORTANT:
        - Always anwser in the same language as the question
        - Be direct and objective in your responses
        - Cite the snippets when relevant
        - If no document is loaded, inform the user
    """,
)

ctx = Context(agent)


async def main():
    print("-" * 30)
    print("🤖 RAG Chatbot - LlamaIndex")
    print("-" * 30)
    print("\nAvailable commands")
    print(" - 'Load <path>' - Load a document")
    print(" - 'exit' - Exit and clean vectordb")
    print("\n💡 Tip: Load a document before making a query")

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
        cleanup_vectordb()
        print("\n 👋 See you later")


if __name__ == "__main__":
    asyncio.run(main())
