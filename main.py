import asyncio
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
collection = client.get_or_create_collection(name="data")

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

reader = DoclingReader()

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
        "./data/Apostila - Agents de IA com Python e LangChain.pdf"
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
