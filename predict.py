import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Class labels (ye tumhara train_generator.class_indices.keys() se milta hai)
class_labels = ['biriyani', 'burger', 'pizza','chocolate_cake', 'cup_cakes'  , 'fries' , 'ice_cream' , 'sandwich','springrolls']  # apne hisaab se adjust karlo

# Model load karo
model = tf.keras.models.load_model('model.h5')

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # image load & resize
    img_array = image.img_to_array(img)                     # image ko array banao
    img_array = img_array / 255.0                            # rescale 1./255 jese training mein kiya tha
    img_array = np.expand_dims(img_array, axis=0)            # batch dimension add karo (1, 224, 224, 3)
    return img_array

def predict_image(img_path):
    img = preprocess_img(img_path)
    preds = model.predict(img)                              # model se prediction lo
    predicted_class = class_labels[np.argmax(preds)]        # sabse zyada probability wali class
    confidence = np.max(preds)                              # uska confidence score
    return predicted_class, confidence

# Example usage
if __name__ == "__main__":
    img_path = "C:\\Users\\Alpha\\Desktop\\im4.jpg"  # yahan apni test image ka path do
    pred_class, conf = predict_image(img_path)
    print(f"Predicted Class: {pred_class}, Confidence: {conf:.2f}")
