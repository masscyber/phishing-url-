import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib

tokenizer = Tokenizer(num_words=5000, char_level=True)
tokenizer.fit_on_texts(["exampleurl.com", "login-paypal.com", "google.com"])
joblib.dump(tokenizer, "tokenizer.pkl")

model = Sequential([
    Embedding(input_dim=5000, output_dim=16, input_length=100),
    LSTM(16),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.save("phishing_url_lstm.h5")
print("✅ Dummy model and tokenizer created successfully!")
