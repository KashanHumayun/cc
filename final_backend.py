import os
import numpy as np
import joblib
import librosa
from flask import Flask, request, jsonify
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import OneHotEncoder
import logging

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the model
def load_model(model_json_path, model_weights_path):
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_path)
    app.logger.info("Loaded model from disk")
    return loaded_model

# Extract features from the audio
def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    
    return result

# Preprocess the voice input
def preprocess_voice_input(audio_path, scaler):
    data, sample_rate = librosa.load(audio_path, duration=2.5, offset=0.6)
    features = extract_features(data, sample_rate)
    features = np.expand_dims(features, axis=0)
    features = scaler.transform(features)
    features = np.expand_dims(features, axis=2)
    return features

# Predict emotion
def predict_emotion(audio_path, model, scaler, encoder):
    processed_input = preprocess_voice_input(audio_path, scaler)
    predictions = model.predict(processed_input)
    predicted_labels = encoder.inverse_transform(predictions)
    return predicted_labels[0][0]  # Extract the single string label

# Load the scaler and encoder
scaler = joblib.load('7scaler.pkl')

# Initialize encoder and fit it with the same labels used during training
encoder = OneHotEncoder()
encoder.fit(np.array(['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']).reshape(-1, 1))

# Load the model
model = load_model('7model.json', '7audio.h5')

@app.route('/upload', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.wav'):
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        try:
            predicted_emotion = predict_emotion(file_path, model, scaler, encoder)
            return jsonify({"emotion": predicted_emotion}), 200
        except Exception as e:
            app.logger.error(f"Error processing file {file.filename}: {str(e)}")
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file format. Only WAV files are accepted."}), 400

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=8000, debug=True)
