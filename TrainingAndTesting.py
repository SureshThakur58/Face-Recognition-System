import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
def load_data(data_directory):
    images = []
    labels = []
    label_map = {}
    current_label = 0

    for person in os.listdir(data_directory):
        person_path = os.path.join(data_directory, person)
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (200, 200))
            images.append(img)

            # Assign numerical labels to names
            if person not in label_map:
                label_map[person] = current_label
                current_label += 1
            labels.append(label_map[person])

    return np.array(images), np.array(labels), label_map

# Prepare data
images, labels, label_map = load_data('data')
images = images / 255.0  # Normalize data
labels = tf.keras.utils.to_categorical(labels)  # One-hot encode labels

# Visualize a few images
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, ax in enumerate(axes):
    index = np.random.randint(0, len(images))
    ax.imshow(images[index])
    ax.set_title(f"Label: {list(label_map.keys())[np.argmax(labels[index])]}")
    ax.axis('off')
plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the model and label map
model.save('face_recognition_model.h5')
np.save('label_map.npy', label_map)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')
