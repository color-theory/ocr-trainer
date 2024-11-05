import csv
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

images = []
labels = []

with open('./data/labels.csv', 'r') as file:
    reader = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        label = row[0]
        image_path = row[1]

        image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale", target_size=(50, 50))
        image = tf.keras.preprocessing.image.img_to_array(image)
        images.append(image)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

images = images / 255.0

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
label_mapping = list(label_encoder.classes_)

with open('label_mapping.json', 'w') as json_file:
    json.dump(label_mapping, json_file)

x_train, x_temp, y_train, y_temp = train_test_split(images, encoded_labels, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=1/3, random_state=42)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=3,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.05
)
datagen.fit(x_train)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(set(labels)), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(datagen.flow(x_train, y_train, batch_size=16), epochs=50, validation_data=(x_val, y_val))
model.export('model_ocr')