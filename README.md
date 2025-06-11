Food Classifier with Urdu and English Recipe and Voice  
CNN-Based AI Project

This AI-powered food recognition project uses a custom CNN model trained from scratch to classify food items from images. After classification, it fetches a detailed recipe and ingredients from a local JSON file, displays them in both English and Urdu, and generates voice output in both languages using Text-to-Speech (TTS).

Key Features

- Custom CNN model built and trained from scratch (not using pre-trained models)  
- Classifies food from an image (e.g., Pizza, Biryani, Salad)  
- Fetches related recipe and ingredients from JSON  
- Displays recipe in both English and Urdu  
- Uses Text-to-Speech to speak recipe in both languages  
- Uses os and tempfile to manage file paths and temporary audio files  
- Offline-ready, lightweight, and expandable  

Model Details

Model Type: Convolutional Neural Network (CNN)  
Framework: TensorFlow / Keras  
Input Size: 224x224  
Classes: Example - Pizza, Burger, Biryani, Salad  
Training: Trained from scratch using custom dataset  
Accuracy: Add your model's accuracy here  

Multi-Language Voice Output (TTS)

This project speaks the food name and recipe in:

- English using pyttsx3 or gTTS  
- Urdu using gTTS with lang='ur'  
- Temporary audio files are created using Python's tempfile library  
- Paths and cleanup handled using the os library  

Example:

from gtts import gTTS  
from playsound import playsound  
import tempfile  
import os  

text_ur = "پیزا ترکیب: چیز اور ٹماٹو ساس شامل کریں"  

with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:  
    tts = gTTS(text_ur, lang='ur')  
    tts.save(temp_audio.name)  
    playsound(temp_audio.name)  
    os.remove(temp_audio.name)  

Sample JSON Recipe File

{
  "Pizza": {
    "english": {
      "ingredients": ["Flour", "Cheese", "Tomato Sauce"],
      "steps": ["Prepare dough", "Add toppings", "Bake in oven"]
    },
    "urdu": {
      "ingredients": ["میدہ", "چیز", "ٹماٹو ساس"],
      "steps": ["آٹا تیار کریں", "ٹاپنگ ڈالیں", "اوون میں بیک کریں"]
    }
  }
}

Project Structure

food-classifier/
  model/
    model.h5                  Trained CNN model
  recipes.json                Contains English and Urdu recipes
  app.py                      Main application (e.g., Streamlit)
  utils/
    tts_handler.py            Voice logic for English and Urdu
  requirements.txt
  README.md
  .gitignore

How to Run

1. Clone the repository:

git clone https://github.com/yourusername/food-classifier.git  
cd food-classifier  

2. Install dependencies:

pip install -r requirements.txt  

3. Run the app:

streamlit run app.py  

Requirements

tensorflow  
numpy  
opencv-python  
gtts  
pyttsx3  
playsound  
streamlit  
os (built-in)  
tempfile (built-in)  

License

This project is released under the MIT License.  
For personal and educational use only. Do not copy or reuse the model for commercial purposes without permission.  

Author

Developer: Your Name  
Email: your@email.com  
GitHub: https://github.com/yourusername  

Disclaimer

This repository does not contain the full dataset or trained model to protect originality.  
Contact the author to request access.
