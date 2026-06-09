import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Dataset path
dataset_path = "dataset"

X = []
y = []

# Read all CSV files
for file in os.listdir(dataset_path):

    if file.endswith(".csv"):

        label = file.split(".")[0]

        file_path = os.path.join(dataset_path, file)

        data = pd.read_csv(file_path, header=None)

        X.extend(data.values)
        y.extend([label] * len(data))

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print("Dataset Loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# One-hot encoding
y_categorical = to_categorical(y_encoded)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42
)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_test, y_test),
    batch_size=16
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Save model
model.save("models/sign_language_model.h5")

# Save labels
np.save("models/labels.npy", encoder.classes_)

print("\nModel saved successfully!")