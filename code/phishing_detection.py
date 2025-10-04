

# 1️⃣ Import Libraries
import os
print("Current Working Directory:", os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D
import shap

# 2️⃣ Load Dataset
csv_path = r"C:\Users\harsh\OneDrive\Desktop\cs A2\data\phishing.csv"
df = pd.read_csv(csv_path)

# 2a️ Check for missing values and fill them
df.fillna(0, inplace=True)

# 2b️ Convert non-numeric columns to numeric
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

# 2c️⃣ Separate features and target
y = df['label'].values
X = df.drop('label', axis=1).values

# 2d️⃣ Reshape for Conv1D
X = X.reshape((X.shape[0], X.shape[1], 1))

# 2e️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Build CNN-LSTM Model
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(2))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nModel Summary:\n")
model.summary()

# 4️⃣ Train Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 5️⃣ Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy*100:.2f}%\n")

# 6️⃣ Plot Accuracy and Loss
import os
if not os.path.exists("../screenshots"):
    os.makedirs("../screenshots")

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("../screenshots/accuracy_plot.png")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("../screenshots/loss_plot.png")
plt.show()

# ===== SHAP Explainability (Fixed for Conv1D input) =====
print("\nCalculating SHAP values for explainability (this may take a few minutes)...")
