import json
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent

model = tf.keras.models.load_model(str(BASE_DIR / "waste_classifier_multiclass.h5"))

with open(BASE_DIR / "class_indices.json", "r", encoding="utf-8") as f:
    class_indices = json.load(f)

class_names = [None] * len(class_indices)
for name, idx in class_indices.items():
    class_names[idx] = name

class_info = {
    "paper": {
        "group": "♻️ Nhóm tái chế",
        "description": "Giấy có thể tái chế thành giấy mới hoặc bao bì.",
    },
    "cardboard": {
        "group": "♻️ Nhóm tái chế",
        "description": "Bìa carton có thể tái chế thành giấy mới hoặc bao bì.",
    },
    "plastic": {
        "group": "♻️ Nhóm tái chế",
        "description": "Nhựa tái chế thành hạt nhựa, sản phẩm nhựa mới.",
    },
    "metal": {
        "group": "♻️ Nhóm tái chế",
        "description": "Kim loại tái chế thành nguyên liệu cho công nghiệp.",
    },
    "white-glass": {
        "group": "♻️ Nhóm tái chế",
        "description": "Thủy tinh trắng tái chế thành chai lọ mới hoặc vật liệu xây dựng.",
    },
    "green-glass": {
        "group": "♻️ Nhóm tái chế",
        "description": "Thủy tinh xanh tái chế thành chai lọ mới hoặc vật liệu xây dựng.",
    },
    "brown-glass": {
        "group": "♻️ Nhóm tái chế",
        "description": "Thủy tinh nâu tái chế thành chai lọ mới hoặc vật liệu xây dựng.",
    },
    "clothes": {
        "group": "♻️ Nhóm tái chế",
        "description": "Quần áo có thể tái sử dụng, quyên góp hoặc tái chế thành sợi.",
    },
    "shoes": {
        "group": "♻️ Nhóm tái chế",
        "description": "Giày dép có thể tái sử dụng, quyên góp hoặc tái chế thành nguyên liệu.",
    },
    "trash": {
        "group": "🗑️ Nhóm rác thải thông thường",
        "description": "Rác không phân loại, thường đem đi chôn lấp hoặc đốt.",
    },
    "biological": {
        "group": "🌱 Nhóm hữu cơ",
        "description": "Rác hữu cơ (thức ăn thừa, lá cây…) có thể làm phân compost hoặc xử lý sinh học.",
    },
    "battery": {
        "group": "⚡ Nhóm nguy hại",
        "description": "Pin cần xử lý riêng để tránh rò rỉ hóa chất độc hại.",
    },
}


def get_full_info(label):
    if label in class_info:
        info = class_info[label]
        return f"{info['group']}\n{info['description']}"
    return "Không có thông tin chi tiết."


last_time = 0
last_frame_processed = None
last_result_text = ""
last_result_full = ""
last_delay = 1.0


def predict_image(img):
    if img is None:
        return "Không có ảnh"

    img_rgb = img if isinstance(img, np.ndarray) else np.array(img)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = img_resized.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    idx = np.argmax(preds)
    label = class_names[idx]
    confidence = preds[idx]
    full_info = get_full_info(label)

    return f"{label} - {confidence:.2%}\n\n{full_info}"


def process_webcam(frame, delay):
    global last_time, last_frame_processed, last_result_text, last_result_full, last_delay

    if frame is None:
        return None, "", "", 0

    current_time = time.time()
    elapsed = current_time - last_time
    remaining = max(0, delay - elapsed)

    if elapsed >= delay or last_frame_processed is None:
        last_time = current_time

        img_resized = cv2.resize(frame, (224, 224))
        img_array = img_resized.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)[0]
        idx = np.argmax(preds)
        label = class_names[idx]
        confidence = preds[idx]

        last_result_text = f"{label} - {confidence:.2%}"
        full_info = get_full_info(label)
        last_result_full = f"{last_result_text}\n\n{full_info}"

        out_frame = frame.copy()
        cv2.putText(
            out_frame,
            f"{label}: {confidence:.2%}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        last_frame_processed = out_frame
        last_delay = delay

        return out_frame, last_result_text, last_result_full, delay

    return last_frame_processed, last_result_text, last_result_full, remaining


def update_timer(delay):
    if last_time == 0:
        return "Chưa bắt đầu"
    elapsed = time.time() - last_time
    remaining = max(0, delay - elapsed)
    return f"Còn {remaining:.1f} giây đến lần phân loại tiếp theo"


with gr.Blocks() as demo:
    gr.Markdown("## Hệ thống phân loại rác thải sinh hoạt")

    with gr.Tab("Webcam trực tiếp"):
        with gr.Row():
            with gr.Column():
                webcam_input = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")
                delay_choice = gr.Dropdown(
                    choices=[0.5, 1.0, 2.0, 5.0, 10.0],
                    value=1.0,
                    label="Thời gian trễ (giây)",
                )
                timer_text = gr.Textbox(label="Thời gian chờ", interactive=False, value="")
            with gr.Column():
                webcam_output = gr.Image(type="numpy", label="Kết quả trên ảnh")
                webcam_result = gr.Textbox(label="Kết quả phân loại (tóm tắt)", lines=2)
                webcam_full = gr.Textbox(label="Chi tiết xử lý", lines=4)

        webcam_input.stream(
            fn=process_webcam,
            inputs=[webcam_input, delay_choice],
            outputs=[webcam_output, webcam_result, webcam_full, timer_text],
            time_limit=0.1,
        )

        timer = gr.Timer(1.0)
        timer.tick(fn=update_timer, inputs=[delay_choice], outputs=[timer_text])

    with gr.Tab("Tải ảnh lên"):
        upload_input = gr.Image(type="numpy", label="Chọn ảnh rác")
        upload_result = gr.Textbox(label="Kết quả phân loại (chi tiết)", lines=6)
        upload_btn = gr.Button("Phân loại")
        upload_btn.click(fn=predict_image, inputs=upload_input, outputs=upload_result)


if __name__ == "__main__":
    demo.launch(share=False)
