import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("models/dog_breed_model.h5")

labels = pd.read_csv("dataset/labels.csv")
encoder = LabelEncoder()
encoder.fit(labels['breed'])

img_path = "dataset/test/0a8d8dda0e354c0571c8d47600ab39a3.jpg"

img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_aaray = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)
confidence = np.max(prediction)*100
predicted_breed = encoder.inverse_transform([predicted_index])
print("Predicted breed:", predicted_breed[0])
print("Confidence: {:.2f}%".format(confidence))