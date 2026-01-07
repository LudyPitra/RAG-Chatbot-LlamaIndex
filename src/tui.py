from textual.app import App, ComposeResult
from textual.widgets import Button, Header, Footer, Static, Input
from textual.containers import Vertical, Horizontal
from textual.binding import Binding
from textual import work
from src.rag_service import agent, ctx, cleanup_vectordb, get_current_document


class MessageBubble(Static):
    """Widget to show a message"""

    def __init__(self, text: str, sender: str = "user") -> None:
        super().__init__(text)
        self.add_class(sender)


class ChatLog(Vertical):
    """Container to show message history"""

    def add_message(self, text: str, sender: str = "user") -> None:
        bubble = MessageBubble(text, sender)
        self.mount(bubble)
        bubble.scroll_visible()


class ChatApp(App):
    CSS = """
       Screen {
        layout: grid;
        grid-size: 1;
        grid-rows: 1fr auto;
    }

    ChatLog {
        height: 1fr;
        border: solid $primary;
        padding: 1;
        overflow-y: auto;
    }

    MessageBubble {
        width: 100%;
        padding: 1;
        margin-bottom: 1;
    }

    MessageBubble.user {
        background: $primary-darken-2;
        border-left: thick $primary;
    }

    MessageBubble.assistant {
        background: $secondary-darken-2;
        border-left: thick $success;
    }

    MessageBubble.system {
        background: $surface;
        border-left: thick $warning;
        text-style: italic;
    }

    #input-area {
        height: auto;
        dock: bottom;
        padding: 1;
    }

    #message-input {
        width: 1fr;
    }

    #send-button {
        width: auto;
        min-width: 10;
    }
   """
    BINDINGS = [
        Binding("f1", "help", "Help"),
        Binding("f2", "load_file", "Load File"),
        Binding("f10", "quit", "Exit"),
        Binding("ctrl+l", "clear", "Clear"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield ChatLog(id="chat-log")

        with Horizontal(id="input-area"):
            yield Input(placeholder="Write...", id="message-input")
            yield Button("Send", id="send-button", variant="primary")

        yield Footer()

    def on_mount(self) -> None:
        """Executed when an application is launched."""
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.add_message(
            "👋 Welcome to the Rag Chatbot\n\n"
            "Commands: \n"
            "• F2 or 'load <path>' - Load Document"
            "• F10 or 'exit' - Close the app"
            "• Ctrl + L - Clean the Chat",
            sender="system",
        )
        self.query_one("#message-input", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """When user press enter to send a message"""
        await self.send_message()

    async def on_buttom_pressed(self, event: Button.Pressed) -> None:
        """When user press the 'send' button"""
        if event.button.id == "send-button":
            await self.send_message()

    async def send_message(self) -> None:
        """Send the message to the agent"""
        input_widget = self.query_one("#message-input", Input)
        message = input_widget.value.strip()

        if not message:
            return

        input_widget.value = ""
        chat_log = self.query_one("#chat-log", ChatLog)

        chat_log.add_message(f"Me: {message}", sender="user")

        self.process_message(message)

    @work(exclusive=True)
    async def process_message(self, message: str) -> None:
        """Process messages with ai agent (background)"""
        chat_log = self.query_one("#chat-log", ChatLog)

        if message.lower().startswith("load "):
            file_path = message[5:].strip()
            chat_log.add_message("⏳ Processing Document...", sender="system")
            response = await agent.run(
                f"Use the load_documents function to load the document: {file_path}",
                ctx=ctx,
            )
        else:
            response = await agent.run(message, ctx=ctx)

        chat_log.add_message(f"Assistant: {response}", sender="assistant")

    def action_help(self):
        """Show help (f1)"""
        chat_log = self.query_one("#chat-log", ChatLog)
        doc = get_current_document()
        doc_info = f"📄Current Document: {doc}" if doc else "📄No Documents Uploaded"

        chat_log.add_message(
            f"ℹ️Help\n\n"
            f"{doc_info}\n\n"
            "Commands:\n"
            "• load <path> - Load Document\n"
            "• F2 - Prompt to load the file\n"
            "• F10 - Exit\n"
            "• Ctrl+L - Clear chat",
            sender="system",
        )

    def action_load_file(self) -> None:
        """Prompt to load a Document (f2)"""
        input_widget = self.query_one("#message-input", Input)
        input_widget.value = "load "
        input_widget.focus()

    def action_clear(self) -> None:
        """Clear the chat (ctrl + L)"""
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.remove_children()
        chat_log.add_message("🧹 Clean chat!", sender="system")

    def on_unmount(self) -> None:
        """Started when the application is closed"""
        cleanup_vectordb()


def main():
    app = ChatApp()
    app.run()


if __name__ == "__main__":
    main()
