import argparse
import asyncio


def launch_main():
    while True:
        print()
        print("╔════════════════════════════════════════╗")
        print("║       🤖 RAG Chatbot - LlamaIndex      ║")
        print("╠════════════════════════════════════════╣")
        print("║                                        ║")
        print("║   Escolha a interface:                 ║")
        print("║                                        ║")
        print("║   [1] CLI - Linha de comando           ║")
        print("║   [2] TUI - Interface gráfica          ║")
        print("║   [0] Sair                             ║")
        print("║                                        ║")
        print("╚════════════════════════════════════════╝")
        print()

        choice = input("Option 👉 ")

        if choice == "1":
            from src.cli import main as cli_main

            asyncio.run(cli_main())
            break
        elif choice == "2":
            from src.tui import main as tui_main

            tui_main()
            break
        elif choice == "0":
            print("\n👋 See you later!")
            break
        else:
            print("\n❌Invalid Option.")


def main():
    parser = argparse.ArgumentParser(
        description="🤖 RAG Chatbot - LlamaIndex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

python main.py --tui      #Launch TUI
python main.py --cli      #Launch CLI
        """,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--cli",
        action="store_true",
        help="Execute CLI mode (Command Line)",
    )

    group.add_argument(
        "--tui",
        action="store_true",
        help="Execute TUI mode (Terminal User Interface)",
    )

    args = parser.parse_args()

    if args.cli:
        from src.cli import main as cli_main

        asyncio.run(cli_main())

    elif args.tui:
        from src.tui import main as tui_main

        tui_main()
    else:
        launch_main()


if __name__ == "__main__":
    main()
