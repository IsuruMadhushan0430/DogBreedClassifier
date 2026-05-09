from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model("models/dog_breed_model.h5")
labels = pd.read_csv("dataset/labels.csv")
encoder = LabelEncoder()
encoder.fit(labels['breed'])

@app.route("/", methods=['GET','POST'])
def home():
    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        file=request.files['image']

        if file:
            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'],
                file.filename
            )
            file.save(filepath)

            img = image.load_img(
                filepath,
                target_size=(224,224)
            )

            img_array = image.img_to_array(img)
            img_array = img_array/255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            pred = model.predict(img_array)
            predicted_index = np.argmax(pred)

            prediction = encoder.inverse_transform(
                [predicted_index]
            )[0]

            confidence = np.max(pred)*100
            image_path = filepath

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )
    
if __name__ == "__main__":
    app.run(debug=True)