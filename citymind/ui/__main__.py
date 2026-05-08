"""CLI entrypoint: `python -m citymind.ui` launches the 3D UI."""

from .app import UIApp


def main() -> None:
    UIApp().run()


if __name__ == "__main__":
    main()
