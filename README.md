# RAG Chatbot - LlamaIndex

A RAG (Retrieval-Augmented Generation) chatbot that allows you to load documents and ask questions about their content. Available in two interfaces: CLI and TUI.

## Features

- Load documents: PDF, DOCX, Markdown, CSV, TXT
- Semantic search within document content
- AI-generated responses based on context
- CLI interface (command line)
- TUI interface (terminal graphics)
- Vector storage with ChromaDB

## Project Structure

```
project/
├── main.py              # Entry point (selection menu)
├── pyproject.toml       # Project dependencies
├── README.md
└── src/
    ├── __init__.py
    ├── rag_agent.py     # RAG agent logic
    ├── cli.py           # CLI interface
    └── tui.py           # TUI interface (Textual)
```

## Installation

### Prerequisites

- Python 3.12+
- Ollama installed and running

### 1. Clone the repository

```bash
git clone https://github.com/LudyPitra/RAG-Chatbot-LlamaIndex.git 
cd rag-chatbot
```

### 2. Install dependencies

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 3. Download Ollama models

```bash
ollama pull ministral-3:14b
ollama pull mxbai-embed-large
```

## How to Use

### Interactive Menu

```bash
python main.py
```

### Direct Access

```bash
# CLI
python main.py --cli

# TUI
python main.py --tui

# Help
python main.py --help
```

## Commands

### CLI and TUI

| Command | Description |
|---------|-------------|
| `load <path>` | Load a document |
| `exit` | Exit the application (CLI) |

### TUI Shortcuts

| Key | Action |
|-----|--------|
| `F1` | Help |
| `F2` | Load document |
| `F10` | Exit |
| `Ctrl+L` | Clear chat |

## Technologies

| Technology | Usage |
|------------|-------|
| LlamaIndex | RAG Framework |
| Ollama | Local LLM |
| ChromaDB | Vector database |
| Textual | TUI interface |
| Docling | Document parser |

## Contributing

1. Fork the project
2. Create a branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request
