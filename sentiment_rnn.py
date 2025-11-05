import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# --- 1. Data Loading and Cleaning ---

# Load the Amazon dataset (rename the downloaded file to this)
try:
    df = pd.read_csv('amazon_reviews.csv', encoding='latin1', header=None, 
                     names=['sentiment_label', 'reviewTitle', 'reviewText'])
    # Select only the relevant columns and drop any rows with missing values
    df = df[['reviewText', 'sentiment_label']].dropna()
    
except FileNotFoundError:
    print("FATAL ERROR: Please download the 'train.csv' file, rename it to 'amazon_reviews.csv', and place it in this directory.")
    # Creating synthetic data structure to allow code to run for demonstration
    df = pd.DataFrame({
        'reviewText': ['great product!', 'terrible quality.', 'it was fine, nothing special', 'worst thing I ever bought', 'Excellent purchase.'],
        'sentiment_label': [2, 1, 2, 1, 2]
    })
    print("Using synthetic data for demonstration.")

# Convert the label column: Class 1 (Negative) becomes 0, Class 2 (Positive) becomes 1
df['sentiment_label'] = df['sentiment_label'].replace({1: 0, 2: 1})

# Basic Text Cleaning Function
def clean_text(text):
    text = str(text).lower() # Ensure it's a string and lowercased
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    return text

df['reviewText'] = df['reviewText'].apply(clean_text)

# Set up features (X) and labels (y)
X = df['reviewText'].values
y = df['sentiment_label'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"--- Data Loaded and Split ---")
print(f"Total Samples: {len(X)}")
print(f"Training Samples: {len(X_train)}")

# --- 2. Tokenization and Padding (Text to Numbers) ---

# Parameters for the tokenizer
MAX_WORDS = 10000 # Max number of unique words to use
MAX_LEN = 100     # Max length of a review sequence

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
tokenizer.fit_on_texts(X_train)

# Convert text to sequences of integers
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure all inputs have the same length
X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"Input Data Shape (Padded): {X_train_padded.shape}")

# --- 3. Build the Deep Learning Model (LSTM) ---

# Vocabulary size is MAX_WORDS
VOCAB_SIZE = MAX_WORDS
EMBEDDING_DIM = 128 # The size of the vector used to represent each word

model = Sequential([
    # Embedding Layer: Maps each word index to a dense vector of size 128
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    
    # LSTM Layer: The core of the RNN, captures sequential dependencies
    LSTM(64, return_sequences=False),
    
    # Dropout: Helps prevent the model from overfitting
    Dropout(0.5),
    
    # Output Layer: Single neuron with sigmoid activation for binary classification (0 or 1)
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\n--- Model Summary ---")
model.summary()

# --- 4. Train the Model ---

# Train the model (use verbose=0 to hide live progress if running in an automated environment)
EPOCHS = 5 # Number of times the model sees the entire dataset
BATCH_SIZE = 32

print(f"\n--- Training Model for {EPOCHS} Epochs ---")
history = model.fit(X_train_padded, y_train, 
                    epochs=EPOCHS, 
                    batch_size=BATCH_SIZE, 
                    validation_data=(X_test_padded, y_test),
                    verbose=1)

# --- 5. Evaluation and Visualization ---

# 5a. Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
print(f"\n--- Final Model Performance ---")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# 5b. Plot Training History (Loss)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('RNN Model Training History (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('rnn_loss_history.png', dpi=300)
plt.show()

print("\nSuccessfully trained RNN model and saved plot to rnn_loss_history.png")