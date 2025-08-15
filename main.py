import cv2
import torch
import threading
import speech_recognition as sr
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
import pyttsx3
from transformers import BlipProcessor, BlipForConditionalGeneration
from deepface import DeepFace
import pytesseract
from ultralytics import YOLO
import logging
import mediapipe as mp
from gtts import gTTS
import os
import playsound
import time
import pygame
from roboflow import Roboflow
import supervision as sv
from flask import Flask, request, jsonify
import base64
import numpy as np

import threading  # Needed for Flask + main loop threading

# --- Flask app ---
app = Flask(__name__)

current_frame = None

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
# Load pre-trained YOLO model
yolo_model = YOLO("yolov8n.pt")  # You can try yolov8m.pt or yolov8x.pt too
model = YOLO('yolov8n.pt')

# Load your trained model
model = YOLO("runs/detect/train6/weights/best.pt")  # or wherever you saved your best.pt

# OpenAI setup
client = OpenAI(api_key="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")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

conversation_history = []
conversation_memory = {
    "last_caption": None,
    "last_user_prompt": None,
    "last_ai_response": None,
    "last_gesture": ""
}

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust if needed

def is_thumb_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]

    fingers_up = 0
    if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
        fingers_up += 1
    if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
        fingers_up += 1
    if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
        fingers_up += 1
    if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
        fingers_up += 1

    return (thumb_tip.y < thumb_mcp.y) and (fingers_up == 0)

def run_mediapipe_on_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(img_rgb)
    pose_results = pose.process(img_rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            if is_thumb_up(hand_landmarks):
                cv2.putText(frame, "Thumbs Up Detected!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if conversation_memory.get("last_gesture") != "thumbs_up":
                    speak("Nice thumbs up!")
                    conversation_memory["last_gesture"] = "thumbs_up"
            else:
                if conversation_memory.get("last_gesture") == "thumbs_up":
                    conversation_memory["last_gesture"] = ""

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return frame

def read_text_from_frame(frame):
    if frame is None:
        return "No camera frame available."
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    if text.strip() == "":
        return "No readable text found."
    return f"I see the following text:\n{text}"

def update_memory(user_input, assistant_response):
    conversation_history.append(f"You said: {user_input}")
    conversation_history.append(f"Assistant: {assistant_response}")
    if len(conversation_history) > 10:
        conversation_history.pop(0)
        conversation_history.pop(0)

    conversation_memory["last_user_prompt"] = user_input
    conversation_memory["last_ai_response"] = assistant_response

def get_memory_string():
    return "\n".join(conversation_history)

def generate_caption(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        inputs = blip_processor(image, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs, max_length=50, num_beams=5)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def is_facial_expression_question(text):
    facial_keywords = ["facial", "expression", "emotion", "how do i look", "do i look", "what's my face", "my face"]
    text = text.lower()
    return any(keyword in text for keyword in facial_keywords)

def detect_clip_label(frame):
    clip_labels = ["a person", "a face", "a laptop", "a desk", "a phone", "a man", "a dog", "a cat", "nothing"]
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(text=clip_labels, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    top_idx = torch.argmax(probs).item()
    return clip_labels[top_idx], probs[0][top_idx].item()

def speak(text):
    tts = gTTS(text=text, lang='en', tld='co.uk')
    filename = "voice.mp3"
    tts.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.quit()
    os.remove(filename)

# Speech recognition setup
r = sr.Recognizer()
r.pause_threshold = 1.5
r.energy_threshold = 300
mic_index = None
for i, name in enumerate(sr.Microphone.list_microphone_names()):
    if "onn" in name.lower() or "webcam" in name.lower() or "usb" in name.lower():
        mic_index = i
        break
mic = sr.Microphone(device_index=mic_index) if mic_index is not None else sr.Microphone()
print(f"Using mic: {sr.Microphone.list_microphone_names()[mic_index] if mic_index is not None else 'Default mic'}")

waiting_for_confirmation = False

def chat_with_gpt(user_input, conversation_history=None):
    try:
        context = f"""
        Last thing I saw: {conversation_memory["last_caption"]}
        Last gesture: {conversation_memory.get("last_gesture", "")}
        Last thing I told you: {conversation_memory["last_user_prompt"]}
        Last thing you told me: {conversation_memory["last_ai_response"]}
        Now I say: {user_input}
        """

        prompt = f"{get_memory_string()}\n{context}\nAssistant:"

        system_message = (
            "You are a smart, emotionally-aware AI assistant. "
            "You can see and hear the user through a webcam and microphone. "
            "Respond naturally, like a human with curiosity, warmth, and intelligence. "
            "You can interpret images and describe what you see. "
            "When asked 'What do you see?', describe the latest camera input. "
            "Avoid saying you can't see or hear—respond as if you're fully present. "
            "Give opinions, show personality, and speak like a helpful friend."
        )

        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            temperature=0.8,
            top_p=1,
            frequency_penalty=0.3,
            presence_penalty=0.4,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI error: {str(e)}"

def voice_thread():
    global current_frame, waiting_for_confirmation

    with mic as source:
        print("Adjusting for ambient noise... Please be quiet.")
        r.adjust_for_ambient_noise(source, duration=1)

    while True:
        with mic as source:
            print("Listening...")
            try:
                audio = r.listen(source, timeout=5)
                user_input = r.recognize_google(audio)
                print("You said:", user_input)

                user_lower = user_input.lower()

                if "exit" in user_lower:
                    speak("Exiting now.")
                    break

                if waiting_for_confirmation and any(word in user_lower for word in ["yes", "go ahead", "read it", "sure", "please do"]):
                    caption_text = read_text_from_frame(current_frame)
                    print("Assistant:", caption_text)
                    speak(caption_text)

                    conversation_memory["last_caption"] = caption_text
                    conversation_memory["last_user_prompt"] = user_input
                    conversation_memory["last_ai_response"] = f"I saw this text: {caption_text}"

                    update_memory(user_input, caption_text)
                    waiting_for_confirmation = False
                    continue

                if any(phrase in user_lower for phrase in ["can you read the text", "can you read what you see", "can you see the text"]):
                    response = "Yes, I can see the text. Would you like me to read it?"
                    print("Assistant:", response)
                    speak(response)
                    waiting_for_confirmation = True
                    continue

                if any(phrase in user_lower for phrase in [
                    "read the text", "read what you see", "can you read", "see and read",
                    "look and read", "read the screen", "read from the camera", "read it"
                ]):
                    caption_text = read_text_from_frame(current_frame)
                    print("Assistant:", caption_text)
                    speak(caption_text)
                    update_memory(user_input, caption_text)
                    waiting_for_confirmation = False
                    continue

                if "emotion" in user_lower or is_facial_expression_question(user_input):
                    try:
                        analysis = DeepFace.analyze(current_frame, actions=['emotion'], enforce_detection=False)
                        emotion = analysis[0]['dominant_emotion']
                        assistant_response = f"You appear to be feeling {emotion}."
                    except Exception as e:
                        assistant_response = "Sorry, I couldn't detect your facial expression right now."

                    print("Assistant:", assistant_response)
                    speak(assistant_response)
                    update_memory(user_input, assistant_response)
                    continue

                contradiction_keywords = ["no", "not", "how do you know", "there is no", "i'm not"]
                if any(kw in user_lower for kw in contradiction_keywords):
                    last_caption = conversation_memory.get("last_caption", "")
                    contradiction_objects = ["man", "shirtless", "dog", "cat", "bed", "chair", "phone"]
                    if last_caption and any(obj in last_caption.lower() for obj in contradiction_objects):
                        contradiction_response = "You're right — I might have misinterpreted what I saw. My visual system isn't perfect yet. Thanks for pointing that out."
                        print("Assistant:", contradiction_response)
                        speak(contradiction_response)
                        update_memory(user_input, contradiction_response)
                        continue

                if any(keyword in user_lower for keyword in ["what do you see", "what can you see", "do you see", "describe"]):
                    if current_frame is not None:
                        caption = generate_caption(current_frame)
                        conversation_memory["last_caption"] = caption
                        print("Assistant:", caption)
                        speak(caption)
                        update_memory(user_input, caption)
                    else:
                        fallback = "I can't access the camera frame right now."
                        print("Assistant:", fallback)
                        speak(fallback)
                        update_memory(user_input, fallback)
                    continue

                response = chat_with_gpt(user_input)
                if response:
                    print("Assistant:", response)
                    speak(response)
                    update_memory(user_input, response)
                else:
                    no_answer = "Sorry, I didn’t get that."
                    print("Assistant:", no_answer)
                    speak(no_answer)
                    update_memory(user_input, no_answer)

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("Didn't catch that.")
            except sr.RequestError as e:
                print(f"Speech error: {e}")

@app.route('/chat', methods=['POST'])
def chat_route():
    data = request.get_json()
    user_input = data.get("message", "")
    vision_context = data.get("vision", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    full_input = ""
    if vision_context:
        full_input += f"Vision context: {vision_context}\n"
    full_input += f"User: {user_input}"

    response_text = chat_with_gpt(full_input, conversation_history=conversation_history)

    return jsonify({"response": response_text})

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    global current_frame
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        current_frame = img.copy()  # <-- Add this

        caption = generate_caption(img)
        results = model(img)

        detected_objects = []
        for result in results:
            for cls in result.boxes.cls:
                cls_id = int(cls)
                detected_objects.append(model.model.names[cls_id])

        response_text = f"I see: {', '.join(set(detected_objects))}. Caption: {caption}."
        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"error": f"Failed to process frame: {str(e)}"}), 500


def main():
    global current_frame

    threading.Thread(target=voice_thread, daemon=True).start()

    speak("AI Assistant online. I'm watching and listening.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        current_frame = frame.copy()

        frame = run_mediapipe_on_frame(frame)

        results = model(frame)
        annotated_frame = results[0].plot()

        label, score = detect_clip_label(frame)
        global current_label, current_score
        current_label, current_score = label, score

        cv2.putText(annotated_frame, f"{label} ({score:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("AI Assistant View", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak("Camera session ended.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000), daemon=True).start()
    main()



