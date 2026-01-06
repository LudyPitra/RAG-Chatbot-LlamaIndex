import sys
import asyncio
import logging
import chromadb
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
    format="%(message)s",
    stream=sys.stdout,
)

logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("ollama").setLevel(logging.ERROR)
logging.getLogger("llama_index").setLevel(logging.ERROR)

async def rag_retrieve(ctx: Context, query: str):
    """Search relevant sections of a document for the agent to use in the response"""
    nodes = retriever.retrieve(query)
    if not nodes:
        return "No relevant excerpt found."

    parts = []

    for i, n in enumerate(nodes, start=1):
        node = getattr(n, "node", n)

        if hasattr(node, "get_content"):
            text = node.get_content()
        elif hasattr(node, "text"):
            text = node.text
        else:
            text = str(node)

        parts.append(f"[{i}]\n{text}")

    return "\n\n".join(parts)


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
    
    supported_extensions = ['.pdf', '.docx', '.md', '.csv', '.txt']
    file_ext = os.path.splittext(file_path)[1].lower()

    if file_ext not in supported_extensions:
        return f"❌ Error: Format '{file_ext}' not supported. Use: {', '.join(supported_extensions)}"

    try:
        print(f"Loading documents: {file_path}")
        print("⏳Processing and creating embeddings...(may take a few seconds")

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

        return f"✅ Document '{current document}' loaded successfully! You can ask your questions."
    
    except Exception as e:
        return f"❌ Error loading document: {str(e)}"


agent = FunctionAgent(
    tools=[rag_retrieve],
    system_prompt="""
    You are a helpful assistant.

    When the user asks anything that may be answered from the loaded documents
    (PDF, DOCX, Markdown, or CSV), ALWAYS call rag_retrieve(query) first to fetch
    relevant excerpts.
    
    After retrieving excerpts:
    - Answer using ONLY the excerpts returned by the tool.
    - Do NOT use prior knowledge or make up information not present in the excerpts.
    - If the tool returns no excerpts, say you couldn't find the answer in the documents
      and ask the user to provide more context (e.g., keywords, section name, or a more
      specific question).

    If the question is not related to the documents, answer normally.
    """,
)

index = VectorStoreIndex.from_documents(
    documents=reader.load_data(
        "./data/A-ARTE-DA-GUERRA.pdf"
    ),
    storage_context=storage_context,
)

retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

ctx = Context(agent)


async def main():
    while True:
        print("\n-----------------------")
        prompt = input("Me: ").strip()

        if not prompt:
            continue

        if prompt.lower() == "exit":
            client.reset()
            break

        response = await agent.run(prompt, ctx=ctx)
        print("\n")
        print(f"Assistant: {str(response)}")


if __name__ == "__main__":
    asyncio.run(main())
