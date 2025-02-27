from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
try:
    with open('model.pkl', 'rb') as f:
        classifier = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("Model files not found! Please run train_model.py first")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json['text']
    # Transform the input text
    text_vectorized = vectorizer.transform([text])
    # Get prediction
    prediction = classifier.predict(text_vectorized)[0]
    
    # Convert prediction to sentiment label
    sentiment = {
        1: "positive",
        0: "neutral",
        -1: "negative"
    }[prediction]
    
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
