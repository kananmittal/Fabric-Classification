import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import pickle

# Load the preprocessed data (from the organizer step)
data = np.load('dataset.npz')
X_train, X_val = data['X_train'], data['X_val']
y_train, y_val = data['y_train'], data['y_val']

# Normalize the images
X_train = X_train / 255.0
X_val = X_val / 255.0

# Build a simple CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification: defect (1) or no defect (0)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=16)

# Save the model
model.save("defect_classifier_model.h5")

# Save training history (optional, for plotting later)
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
