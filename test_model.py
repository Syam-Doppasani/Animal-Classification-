import tensorflow as tf
import numpy as np

# Load model and class names
model = tf.keras.models.load_model("animal_classifier_model.h5")
with open("class_names.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

# Predict
img_path = "test.jpg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_batch)
pred_index = np.argmax(predictions[0])
confidence = predictions[0][pred_index]

print("🔎 Predicted:", class_names[pred_index])
print("📈 Confidence:", round(confidence * 100, 2), "%")
