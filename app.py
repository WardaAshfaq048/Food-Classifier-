import streamlit as st
import tensorflow as tf
from PIL import Image 
import numpy as np
import json
from googletrans import Translator
from gtts import gTTS
import os
import tempfile

# Load model and data
model = tf.keras.models.load_model('model.h5')

food_types = ['biriyani', 'burger', 'chocolate_cake', 'cup_cakes', 'fries', 
             'ice_cream', 'pizza', 'sandwich', 'springrolls']


def load_recipes():
    with open('recipe.json', 'r') as f:
        return json.load(f)

recipes = load_recipes()
translator = Translator()

# App title
st.title("🍽️ Food Classifier with Recipes")
st.write("Upload a food image to get recipes in English/Urdu with voice")

# Image processing
def prepare_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Translation and voice functions
def translate(text, to_urdu=True):
    try:
        return translator.translate(text, dest='ur' if to_urdu else 'en').text
    except:
        return text

def text_to_speech(text, urdu=False):
        tts = gTTS(text=text, lang='ur' if urdu else 'en')
        audio_file = os.path.join(tempfile.gettempdir(), "recipe_audio.mp3")
        tts.save(audio_file)
        st.audio(audio_file)

# Main app flow
uploaded_file = st.file_uploader("📤 Upload food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Your Food", width=300)
    
    # Prediction
    img_array = prepare_image(img)
    prediction = model.predict(img_array)
    food_name = food_types[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Food: {food_name} — Confidence: {confidence:.1f}%")

    # Find first matching recipe
    recipe = next((r for r in recipes if r["category"] == food_name), None)

    if recipe:
        st.subheader(f"🍴 {recipe['name']}")
        show_urdu = st.checkbox("Show in Urdu", key=f"urdu_{recipe['name']}")

        full_text = ""

        # Ingredients
        st.write("### 🧂 Ingredients:" if not show_urdu else "### 🧂 اجزاء:")
        for item, amount in recipe["ingredients"].items():
            text_line = f"{item.replace('_', ' ')}: {amount}"
            if show_urdu:
                text_line = translate(text_line)
            st.write(f"- {text_line}")
            full_text += text_line + "\n"

        # Steps
        st.write("### 👩‍🍳 Steps:" if not show_urdu else "### 👩‍🍳 ترکیب:")
        for i, step in enumerate(recipe["steps"], 1):
            if show_urdu:
                step = translate(step)
            st.write(f"{i}. {step}")
            full_text += step + "\n"

        # Voice button
        if st.button("🔊 Listen to Recipe"):
            text_to_speech(full_text, urdu=show_urdu)
    else:
        st.warning("No recipe found for this food.")
