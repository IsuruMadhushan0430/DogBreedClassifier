import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

labels = pd.read_csv("dataset/labels.csv")
labels['image_path'] = labels['id'].apply(
    lambda x: "dataset/train/"+x+".jpg"
)

encoder = LabelEncoder()
labels['breed_encoded'] = encoder.fit_transform(labels['breed'])

num_classes = len(labels['breed'].unique())

IMG_SIZE = 224

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0
    return image, label

dataset = tf.data.Dataset.from_tensor_slices(
    (
        labels['image_path'].values,
        labels['breed_encoded'].values
    )
)

dataset = dataset.map(load_image)
dataset  = dataset.batch(32)
dataset = dataset.shuffle(1000)

#AI model building

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(120, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(dataset, epochs=10)

model.save("dog_breed_model.h5")