import socket
import threading
import time
import webbrowser

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}/"


def is_server_up():
    try:
        with socket.create_connection((HOST, PORT), timeout=0.5):
            return True
    except OSError:
        return False


def open_browser_when_ready():
    for _ in range(180):
        if is_server_up():
            webbrowser.open(URL)
            return
        time.sleep(0.5)
    print(f"Không mở được trình duyệt. Hãy vào thủ công: {URL}")


if __name__ == "__main__":
    if is_server_up():
        print(f"Server đang chạy sẵn tại {URL}, chỉ mở trình duyệt.")
        webbrowser.open(URL)
    else:
        import uvicorn
        from game_api import app

        threading.Thread(target=open_browser_when_ready, daemon=True).start()
        print("Đang khởi động Chiến Binh Xanh... (lần đầu tải model có thể mất ~10 giây)")
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")
