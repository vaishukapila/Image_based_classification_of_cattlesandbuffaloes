import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS

# --- TENSORFLOW IMPORTS ---
import tensorflow as tf
import numpy as np

# --- OPENCV IMPORT ---
import cv2
import time


# -------------------- CUSTOM CSS --------------------
def load_css():
    st.markdown("""
    <style>
    /* Colorful Agricultural Background */
    .stApp {
        background: 
            linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, rgba(52, 152, 219, 0.1) 25%, 
                           rgba(155, 89, 182, 0.1) 50%, rgba(241, 196, 15, 0.1) 75%, 
                           rgba(231, 76, 60, 0.1) 100%),
            url("https://png.pngtree.com/thumb_back/fh260/background/20240919/pngtree-a-background-of-orange-blue-and-yellow-gradients-with-gritty-appearance-image_16233934.jpg") center center fixed;
        background-size: cover;
        background-blend-mode: overlay;
    }

    /* Enhanced backdrop overlay for better contrast */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 20%, rgba(46, 204, 113, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(52, 152, 219, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 40% 60%, rgba(155, 89, 182, 0.1) 0%, transparent 50%),
            linear-gradient(45deg, rgba(241, 196, 15, 0.05) 0%, rgba(231, 76, 60, 0.05) 100%);
        pointer-events: none;
        z-index: -1;
    }

    /* Enhanced text colors for better readability */
    body, .stMarkdown, .stText, p, span, div { 
        color: #2c3e50 !important; 
        font-weight: 500 !important;
        text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8) !important;
    }

    .card, .info-message, .upload-section, .voice-input-display { 
        color: #2c3e50 !important; 
        font-weight: 500 !important;
    }

    .stChatMessage { 
        color: #2c3e50 !important; 
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
    }

    .subtitle { 
        color: #34495e !important; 
        font-weight: 600 !important;
        text-shadow: 0 2px 4px rgba(255, 255, 255, 0.8) !important;
    }

    .stTextInput > div > div > input { 
        color: #2c3e50 !important; 
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(52, 152, 219, 0.3) !important;
        font-weight: 500 !important;
    }

    .stButton > button { 
        color: #fff !important; 
        font-weight: 600 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    }

    /* Enhanced main title with agricultural theme */
    .main-title {
        background: linear-gradient(45deg, #27AE60, #2ECC71, #F39C12, #E67E22, #8E44AD, #3498DB);
        background-size: 400% 400%;
        animation: gradient 6s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
        filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        color: #2c3e50 !important;
        background: rgba(255, 255, 255, 0.8);
        padding: 10px 20px;
        border-radius: 25px;
        display: inline-block;
        backdrop-filter: blur(5px);
    }

    /* Enhanced card design */
    .card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 250, 0.95) 100%);
        padding: 30px;
        border-radius: 25px;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.1),
            0 8px 16px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        margin: 15px;
        border: 1px solid rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
    }

    /* Success message with agricultural colors */
    .success-message {
        background: linear-gradient(45deg, #27AE60, #2ECC71);
        color: white;
        padding: 18px;
        border-radius: 20px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 25px 0;
        box-shadow: 0 8px 16px rgba(39, 174, 96, 0.3);
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }

    /* Voice input display */
    .voice-input-display {
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.15) 0%, rgba(155, 89, 182, 0.15) 100%);
        padding: 15px;
        border-radius: 20px;
        margin: 12px 0;
        border: 2px solid rgba(52, 152, 219, 0.3);
        backdrop-filter: blur(5px);
        color: #2c3e50 !important;
    }

    /* Enhanced breed tags */
    .breed-tag {
        background: linear-gradient(45deg, #3498DB, #2980B9, #8E44AD);
        padding: 12px 16px;
        margin: 5px;
        border-radius: 20px;
        text-align: center;
        font-size: 0.9rem;
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease;
    }

    .breed-tag:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(52, 152, 219, 0.4);
    }

    /* Enhanced info boxes */
    .stInfo {
        background: rgba(52, 152, 219, 0.1) !important;
        border: 2px solid rgba(52, 152, 219, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(5px) !important;
    }

    .stSuccess {
        background: rgba(46, 204, 113, 0.1) !important;
        border: 2px solid rgba(46, 204, 113, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(5px) !important;
    }

    .stError {
        background: rgba(231, 76, 60, 0.1) !important;
        border: 2px solid rgba(231, 76, 60, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(5px) !important;
    }

    .stWarning {
        background: rgba(241, 196, 15, 0.1) !important;
        border: 2px solid rgba(241, 196, 15, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(5px) !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 20px !important;
        padding: 10px !important;
        backdrop-filter: blur(10px) !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }

    /* File uploader styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 15px !important;
        padding: 15px !important;
        border: 2px dashed rgba(52, 152, 219, 0.4) !important;
        backdrop-filter: blur(5px) !important;
    }

    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
        text-shadow: 0 2px 4px rgba(255, 255, 255, 0.8) !important;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 250, 0.95) 100%) !important;
        backdrop-filter: blur(10px) !important;
    }
    </style>
    """, unsafe_allow_html=True)


# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Cattle & Buffalo Breed Identifier + AI Assistant", layout="wide")
load_css()
st.markdown('<h1 class="main-title">🐄 Cattle & Buffalo Breed Identification</h1>', unsafe_allow_html=True)



# -------------------- GEMINI API --------------------
API_KEY = ""  # Add your API key here
if not API_KEY:
    st.error("⚠ Please add your Google AI API key in the API_KEY variable.")
    st.stop()
else:
    genai.configure(api_key=API_KEY)

# -------------------- BREED MODEL CLASSES & DATA --------------------
CLASS_NAMES = [
    'Brown_Swiss', 'Deoni', 'Gir', 'Holstein_Friesian', 'Jaffrabadi',
    'Kangayam', 'Kankrej', 'Khillari', 'Murrah', 'Pandharpuri', 'Sahiwal', 'Toda'
]

BREED_ADVISORY = {
    "Brown_Swiss": "Brown Swiss are good for both milk and draught. Ensure high-quality fodder and proper shelter.",
    "Deoni": "Deoni breed is drought-resistant. Provide mineral-rich feed and ensure vaccination for common diseases.",
    "Gir": "Gir cattle are famous for high milk yield. Maintain clean water supply and fodder with green grass.",
    "Holstein_Friesian": "HF cows need proper cooling in hot climates. Provide balanced feed with high protein.",
    "Jaffrabadi": "Jaffrabadi buffaloes need regular bathing to stay cool. Provide oilseed cakes for better milk production.",
    "Kangayam": "Kangayam cattle are hardy draught animals. Provide dry fodder and supplements during peak work.",
    "Kankrej": "Kankrej are dual-purpose. Ensure vaccination and disease prevention, especially foot and mouth disease.",
    "Khillari": "Khillari breed thrives in dry conditions. Provide jaggery and mineral mixture for energy.",
    "Murrah": "Murrah buffaloes need high-quality fodder, oilseed cakes, and proper wallowing ponds for cooling.",
    "Pandharpuri": "Pandharpuri buffaloes are hardy. Provide seasonal green fodder and ensure clean drinking water.",
    "Sahiwal": "Sahiwal cattle are heat-tolerant. Provide sufficient shade and maintain tick control measures.",
    "Toda": "Toda breed is rare and native to Nilgiris. Provide forest-based fodder and traditional care methods."
}


# -------------------- LOAD PYTORCH BREED MODEL --------------------
@st.cache_resource
def load_breed_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    model.load_state_dict(torch.load('cattle_buffalo_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


breed_model = load_breed_model()
breed_image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_breed(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB": image = image.convert("RGB")
    tensor = breed_image_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = breed_model(tensor)
        _, idx = torch.max(outputs, 1)
    return CLASS_NAMES[idx.item()]


# --- LOAD TENSORFLOW SKIN DISEASE MODEL ---
@st.cache_resource
def load_skin_model():
    model = tf.keras.models.load_model('skin_model.h5')
    return model


skin_model = load_skin_model()


# ---------- Robust preprocessing for skin_model ----------
def prepare_image_for_skin_model(image_bytes):
    """
    Prepares an image to the exact input expected by skin_model.
    Handles:
      - models with 4D inputs: (None, H, W, C)
      - models with flat inputs: (None, N)  -> we will resize, flatten and pad/truncate to N
      - grayscale expectations
    Returns: numpy array shaped correctly for model.predict(...)
    """
    img = Image.open(io.BytesIO(image_bytes))
    # ensure RGB initially
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    model_input_shape = skin_model.input_shape  # possibly tuple like (None, h, w, c) or (None, N)
    # if model_input_shape is a list (multi-input), take first
    if isinstance(model_input_shape, list):
        model_input_shape = model_input_shape[0]

    # Drop batch dim
    if len(model_input_shape) == 4:
        _, H, W, C = model_input_shape
        # fallback sizes if None
        H = H or 224
        W = W or 224
        C = C or 3

        # If model expects single channel but image is RGB, convert
        if C == 1:
            img = img.convert("L")
        else:
            if img.mode == "L":
                img = img.convert("RGB")

        img = img.resize((W, H))
        arr = tf.keras.preprocessing.image.img_to_array(img).astype("float32") / 255.0

        # ensure channel dimension matches
        if arr.shape[2] != C:
            # convert channels appropriately
            if C == 1:
                arr = np.expand_dims(np.mean(arr, axis=2), axis=2)  # rgb -> gray by average
            elif C == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)

        arr = np.expand_dims(arr, axis=0)  # add batch
        return arr

    elif len(model_input_shape) == 2:
        # Flatten input expected: (None, N)
        N = model_input_shape[1]
        # Try to assume 3 channels unless model expects 1
        # We'll attempt to resize to a square and then flatten, padding/truncating to N.
        # Choose 3 channels if image is RGB else 1
        target_c = 3 if img.mode == "RGB" else 1
        # compute side length
        side = int(round(np.sqrt(N / target_c)))
        if side < 1:
            side = max(1, int(np.sqrt(N)))  # fallback

        # resize and get array
        if target_c == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img = img.resize((side, side))
        arr = tf.keras.preprocessing.image.img_to_array(img).astype("float32") / 255.0
        flat = arr.flatten()
        L = flat.shape[0]

        if L > N:
            flat = flat[:N]
        elif L < N:
            pad = np.zeros((N - L,), dtype=flat.dtype)
            flat = np.concatenate([flat, pad], axis=0)

        flat = np.expand_dims(flat, axis=0)  # batch
        return flat

    else:
        # Unexpected shape length; try to coerce: resize to 224x224x3 and hope for the best
        img = img.convert("RGB").resize((224, 224))
        arr = tf.keras.preprocessing.image.img_to_array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr


def predict_skin_disease(image_bytes):
    """
    Uses the robust prepare_image_for_skin_model() to adapt input to model.
    """
    try:
        arr = prepare_image_for_skin_model(image_bytes)
        prediction = skin_model.predict(arr)
        # handle binary-like outputs (assume sigmoid at final node) OR class probabilities
        if prediction.ndim == 2 and prediction.shape[1] == 1:
            prob = float(prediction[0][0])
            return ("Lumpy Skin Disease Detected" if prob > 0.5 else "Healthy Skin"), prob
        elif prediction.ndim == 2 and prediction.shape[1] >= 2:
            # assume [healthy_prob, disease_prob] or similar
            disease_index = 1  # assume class 1 is disease; adjust if your model differs
            prob = float(prediction[0][disease_index])
            return ("Lumpy Skin Disease Detected" if prob > 0.5 else "Healthy Skin"), prob
        else:
            # fallback: interpret highest class as disease if index 1
            idx = np.argmax(prediction, axis=1)[0]
            if idx == 1:
                return "Lumpy Skin Disease Detected", None
            else:
                return "Healthy Skin", None
    except Exception as e:
        # return error message as string for display
        return f"Error during skin model prediction: {e}", None


# -------------------- GEMINI AI --------------------
def get_gemini_response(breed, question):
    prompt = f"You are an expert on Indian cattle and buffalo breeds.\nBreed: '{breed}'\nQuestion: '{question}'"
    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash-latest")
        return model_gemini.generate_content(prompt).text
    except Exception as e:
        return f"AI service unavailable. Error: {e}"


# -------------------- VOICE FUNCTIONS --------------------
def speech_to_text(lang='en-US'):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            st.success("Audio captured!")
            return recognizer.recognize_google(audio, language=lang)
        except:
            st.warning("Could not understand audio.")
            return None


def text_to_speech(text, lang='en'):
    if not text: return
    try:
        tts = gTTS(text=text, lang=lang)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        st.audio(audio_fp, format='audio/mp3')
    except:
        st.error("Failed to generate audio.")


# -------------------- SESSION STATE --------------------
if "history" not in st.session_state: st.session_state.history = []
if "messages" not in st.session_state: st.session_state.messages = []
if "prediction" not in st.session_state: st.session_state.prediction = None
# camera run state
if "run_camera" not in st.session_state: st.session_state.run_camera = False

# -------------------- TABS --------------------
tab_home, tab_skin, tab_advisory, tab_ai, tab_history, tab_live = st.tabs(
    ["🏠 Home (Breed ID)", "🩺 Skin Disease Detection", "🌾 Farmer Advisory", "💬 AI Assistant",
     "📊 History & Analytics", "📷 Live Camera Detection"]
)

# -------------------- HOME TAB --------------------
with tab_home:
    st.header("Upload Image & Predict Breed")
    uploaded_file_breed = st.file_uploader("📸 Upload an image of cattle or buffalo", type=["jpg", "jpeg", "png"],
                                           key="breed_uploader")
    if uploaded_file_breed:
        image_bytes = uploaded_file_breed.getvalue()
        prediction = predict_breed(image_bytes)
        st.session_state.prediction = prediction
        st.session_state.history.append(prediction)

        st.image(image_bytes, use_column_width=True, caption="Uploaded Image")
        st.success(f"🎯 Predicted Breed: {prediction.replace('_', ' ')}")

        st.subheader("🐮 Supported Breeds")
        cols = st.columns(2)
        for i, breed in enumerate(CLASS_NAMES):
            col = cols[i % 2]
            with col: st.markdown(f'<div class="breed-tag">{breed.replace("_", " ")}</div>', unsafe_allow_html=True)
    else:
        st.info("Please upload an image to get a breed prediction.")

# -------------------- SKIN DISEASE TAB --------------------
with tab_skin:
    st.header("Detect Lumpy Skin Disease")
    uploaded_file_skin = st.file_uploader("📸 Upload an image of cow skin", type=["jpg", "jpeg", "png"],
                                          key="skin_uploader")
    if uploaded_file_skin:
        image_bytes_skin = uploaded_file_skin.getvalue()

        with st.spinner('Analyzing skin condition...'):
            skin_prediction, prob = predict_skin_disease(image_bytes_skin)

        st.image(image_bytes_skin, use_column_width=True, caption="Uploaded Skin Image")

        if isinstance(skin_prediction, str) and skin_prediction.startswith("Error"):
            st.error(skin_prediction)
            st.info(
                "Possible cause: model expects a different input shape. Check model.input_shape and retrain or adjust preprocessing accordingly.")
        else:
            if "Healthy" in skin_prediction:
                if prob is not None:
                    st.success(f"✅ Result: {skin_prediction} (prob {prob:.2f})")
                else:
                    st.success(f"✅ Result: {skin_prediction}")
            else:
                if prob is not None:
                    st.error(f"⚠ Result: {skin_prediction} (prob {prob:.2f})")
                else:
                    st.error(f"⚠ Result: {skin_prediction}")
                st.warning("Advisory: Please consult a veterinarian immediately for confirmation and treatment.")
    else:
        st.info("Please upload an image of the animal's skin to check for Lumpy Skin Disease.")

# -------------------- ADVISORY TAB --------------------
with tab_advisory:
    st.header("Farmer Advisory for Breeds")
    if st.session_state.prediction:
        st.subheader(f"Advice for {st.session_state.prediction.replace('_', ' ')}")
        st.info(BREED_ADVISORY.get(st.session_state.prediction, "No advisory available for this breed."))
    else:
        st.info("Make a breed prediction first in the Home tab to see an advisory.")

# -------------------- AI ASSISTANT TAB --------------------
with tab_ai:
    st.header("AI Chat Assistant")
    if not st.session_state.prediction:
        st.info("Make a breed prediction first in the Home tab to ask questions about that breed.")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_question = None
        if st.button("🎤 Ask with Voice"):
            user_question = speech_to_text()
        text_input = st.chat_input(f"Or type your question about {st.session_state.prediction.replace('_', ' ')}...")
        if text_input: user_question = text_input

        if user_question:
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"): st.markdown(user_question)

            with st.spinner("AI is thinking..."):
                response = get_gemini_response(st.session_state.prediction, user_question)

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
                text_to_speech(response)
            st.rerun()

# -------------------- HISTORY & ANALYTICS TAB --------------------
with tab_history:
    st.header("Prediction History & Analytics")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history, columns=["Breed"])
        st.write("### Breed Prediction History")
        st.dataframe(df)

        st.write("### Prediction Counts")
        breed_counts = df["Breed"].value_counts()
        st.bar_chart(breed_counts)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download History as CSV", csv, "prediction_history.csv", "text/csv")
    else:
        st.info("No breed predictions have been made yet.")

# -------------------- LIVE CAMERA DETECTION TAB --------------------
with tab_live:
    st.header("📷 Live Camera Detection")
    # Use session_state key so we can change it during runtime
    st.checkbox("Start Camera", key="run_camera")
    if st.session_state.run_camera:
        cap = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([])

        st.info("Camera started. Uncheck 'Start Camera' to stop.")

        try:
            while st.session_state.run_camera and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video frame.")
                    break

                # Convert OpenCV BGR to RGB for PIL/display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                # Save to buffer
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG")
                img_bytes = buf.getvalue()

                # Breed prediction (fast)
                try:
                    breed_pred = predict_breed(img_bytes)
                except Exception as e:
                    breed_pred = f"Breed model error: {e}"

                # Skin prediction (robust)
                skin_pred, skin_prob = predict_skin_disease(img_bytes)

                # Put overlay text onto image for display (use RGB array)
                display_frame = rgb_frame.copy()
                cv2.putText(display_frame, f"Breed: {breed_pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2, cv2.LINE_AA)
                if isinstance(skin_pred, str) and skin_pred.startswith("Error"):
                    cv2.putText(display_frame, "Skin: Error", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    label = skin_pred if skin_prob is None else f"{skin_pred} ({skin_prob:.2f})"
                    cv2.putText(display_frame, f"Skin: {label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 0, 0), 2, cv2.LINE_AA)

                FRAME_WINDOW.image(display_frame)

                # tiny sleep for UI
                time.sleep(0.08)
        except Exception as e:
            st.error(f"Camera loop error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    else:
        st.info("Camera stopped. Check the box to start the webcam.")
