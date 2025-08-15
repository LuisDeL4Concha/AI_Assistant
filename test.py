from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import cv2
import numpy as np
from queue import Queue
from main import chat_with_gpt, generate_caption, model

app = Flask(__name__)
CORS(app)

conversation_history = []
frame_queue = Queue(maxsize=1)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    from speech_recognition import Recognizer, AudioFile
    file = request.files['audio']
    file.save("temp.wav")
    recognizer = Recognizer()
    with AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return jsonify({"transcript": text})

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        img_data = data.get("image","")
        if not img_data:
            return jsonify({"error":"No image data provided"}), 400

        if "," in img_data:
            img_data = img_data.split(",")[1]

        image_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Only keep latest frame
        if not frame_queue.empty():
            try: frame_queue.get_nowait()
            except: pass
        frame_queue.put(frame)

        latest_frame = frame_queue.get()
        results = model(latest_frame)
        yolo_detections = [model.model.names[int(cls)] for r in results for cls in r.boxes.cls]
        caption = generate_caption(latest_frame)
        description = f"I see: {', '.join(set(yolo_detections))}. Caption: {caption}."
        return jsonify({"response": description})

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history
    data = request.get_json()
    user_input = data.get("message","")
    vision_context = data.get("vision","")

    if not user_input:
        return jsonify({"error":"No message provided"}),400

    full_input = ""
    if vision_context:
        full_input += f"Vision context: {vision_context}\n"
    full_input += f"User: {user_input}"

    conversation_history.append({"role":"user","content":full_input})
    response = chat_with_gpt(full_input, conversation_history=conversation_history)
    conversation_history.append({"role":"assistant","content":response})

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=5000)
