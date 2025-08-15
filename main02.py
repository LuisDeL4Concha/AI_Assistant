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

# === OpenAI Setup ===
client = OpenAI(api_key="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")

# === Device Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load CLIP ===
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === Load BLIP ===
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Conversation memory (limit to last 10 messages)
conversation_history = []

conversation_memory = {
    "last_caption": None,
    "last_user_prompt": None,
    "last_ai_response": None
}
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Change this path if needed

def read_text_from_camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "I couldn't access your camera."

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    if text.strip():
        return f"I see this text: {text.strip()}"
    else:
        return "I couldn't detect any readable text."


def update_memory(user_input, assistant_response):
    conversation_history.append(f"You said: {user_input}")
    conversation_history.append(f"Assistant: {assistant_response}")
    if len(conversation_history) > 10:
        conversation_history.pop(0)
        conversation_history.pop(0)

    # Update short-term memory
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

def detect_emotion_from_camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "I couldn't access your camera."
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion']
        return f"You appear to be feeling {dominant_emotion}."
    except Exception as e:
        return f"Sorry, I couldn't detect your facial expression. ({str(e)})"

def speak(text):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if "zira" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 175)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("SPEAK ERROR:", e)

# === Microphone Setup ===
r = sr.Recognizer()
mic_index = None
for i, name in enumerate(sr.Microphone.list_microphone_names()):
    if "onn" in name.lower() or "webcam" in name.lower() or "usb" in name.lower():
        mic_index = i
        break
mic = sr.Microphone(device_index=mic_index) if mic_index is not None else sr.Microphone()
print(f"Using mic: {sr.Microphone.list_microphone_names()[mic_index] if mic_index is not None else 'Default mic'}")

# === CLIP Labels ===
clip_labels = ["a person", "a face", "a laptop", "a desk", "a phone", "a man", "a woman", "a dog", "a cat", "nothing"]

def detect_clip_label(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(text=clip_labels, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    top_idx = torch.argmax(probs).item()
    return clip_labels[top_idx], probs[0][top_idx].item()

def chat_with_gpt(user_input):
    try:
        context = f"""
        Last thing I saw: {conversation_memory["last_caption"]}
        Last user said: {conversation_memory["last_user_prompt"]}
        Last assistant response: {conversation_memory["last_ai_response"]}
        Now user says: {user_input}
        """

        prompt = f"{get_memory_string()}\n{context}\nAssistant:"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're a helpful AI assistant watching and listening to the user through a webcam."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI error: {str(e)}"

# === Global Shared State ===
current_label = ""
current_score = 0.0
current_frame = None

# === Voice Thread ===
def voice_thread():
    global current_label, current_score, current_frame
    with mic as source:
        r.adjust_for_ambient_noise(source)
    while True:
        with mic as source:
            print("Listening...")
            try:
                audio = r.listen(source, timeout=5)
                user_input = r.recognize_google(audio)
                print("You said:", user_input)

                # 🔍 CONTRADICTION LOGIC START
                contradiction_keywords = ["no", "not", "how do you know", "there is no", "i'm not"]
                if any(kw in user_input.lower() for kw in contradiction_keywords):
                    last_caption = conversation_memory.get("last_caption", "")
                    contradiction_objects = ["man", "woman", "shirtless", "dog", "cat", "bed", "chair", "phone"]
                    if last_caption and any(obj in last_caption.lower() for obj in contradiction_objects):
                        contradiction_response = "You're right — I might have misinterpreted what I saw. My visual system isn't perfect yet. Thanks for pointing that out."
                        print("Assistant:", contradiction_response)
                        speak(contradiction_response)
                        update_memory(user_input, contradiction_response)
                        conversation_memory["last_user_prompt"] = user_input
                        conversation_memory["last_ai_response"] = contradiction_response
                        continue  # 🔁 SKIP GPT for this round
                # 🔍 CONTRADICTION LOGIC END

                if "exit" in user_input.lower():
                    speak("Exiting now.")
                    break
                if "read" in user_input.lower() and "text" in user_input.lower():
                    response = read_text_from_camera()
                    print("Assistant:", response)
                    speak(response)
                    update_memory(user_input, response)

                elif is_facial_expression_question(user_input):
                    expression_response = detect_emotion_from_camera()
                    print("Assistant:", expression_response)
                    speak(expression_response)
                    update_memory(user_input, expression_response)

                elif "what do you see" in user_input.lower() or "what can you see" in user_input.lower():
                    if current_frame is not None:
                        caption = generate_caption(current_frame)
                        print("Caption:", caption)
                        speak(caption)
                        update_memory(user_input, caption)

                        # Save to short-term logic memory
                        conversation_memory["last_caption"] = caption
                        conversation_memory["last_user_prompt"] = user_input
                        conversation_memory["last_ai_response"] = caption

                    else:
                        speak("I can't see anything right now.")
                        update_memory(user_input, "I can't see anything right now.")

                elif "do you see" in user_input.lower():
                    if current_label:
                        label_response = f"I see {current_label} with {current_score:.2f} confidence."
                        speak(label_response)
                        update_memory(user_input, label_response)
                    else:
                        speak("I'm still looking.")
                        update_memory(user_input, "I'm still looking.")

                else:
                    answer = chat_with_gpt(user_input)
                    if answer:
                        print("Assistant:", answer)
                        speak(answer)
                        update_memory(user_input, answer)
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

# === Start Voice Thread ===
threading.Thread(target=voice_thread, daemon=True).start()

# === Camera Feed Loop ===
speak("AI Assistant online. I'm watching and listening.")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = frame.copy()
    label, score = detect_clip_label(frame)
    current_label, current_score = label, score

    cv2.putText(frame, f"{label} ({score:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("AI Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        speak("Camera session ended.")
        break

cap.release()
cv2.destroyAllWindows()