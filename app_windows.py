import socket
import threading
import time
import webbrowser
from pathlib import Path

try:
    import webview
except ImportError:
    webview = None

from app_multiclass import demo

BASE_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 7861
URL = f"http://{HOST}:{PORT}"


def run_gradio():
    demo.launch(server_port=PORT, quiet=True)


def wait_until_ready(timeout=120):
    for _ in range(timeout * 2):
        try:
            with socket.create_connection((HOST, PORT), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.5)
    return False


if __name__ == "__main__":
    threading.Thread(target=run_gradio, daemon=True).start()

    if not wait_until_ready():
        raise SystemExit("Không khởi động được Gradio server.")

    if webview is not None:
        webview.create_window(
            title="Hệ thống phân loại rác thải AI",
            url=URL,
            width=1280,
            height=720,
            resizable=True,
            fullscreen=False,
            min_size=(800, 600),
        )
        webview.start(icon=str(BASE_DIR / "Logo.ico"))
    else:
        webbrowser.open(URL)
        input("Nhấn Enter để thoát...")
