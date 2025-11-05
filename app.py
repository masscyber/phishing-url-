from flask import Flask, render_template, request
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = tf.keras.models.load_model("phishing_url_lstm.h5")
tokenizer = joblib.load("tokenizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    url = request.form["url"]
    seq = tokenizer.texts_to_sequences([url])
    pad = pad_sequences(seq, maxlen=100)
    pred = model.predict(pad)[0][0]
    label = "Phishing" if pred > 0.5 else "Legitimate"
    return render_template("index.html",
                           prediction_text=f"The URL is {label} ({pred:.2f})")

if __name__ == "__main__":
    app.run(debug=True)
