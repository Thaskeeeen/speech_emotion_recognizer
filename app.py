import os
import joblib
import numpy as np
import soundfile as sf
import librosa
import tempfile
import pydub
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import requests


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB
app.config['UPLOAD_FOLDER'] = 'C:\\Users\\hp\\Desktop\\separate folder\\uploaded_audio'

# Load the ML model
model = joblib.load('final1.pkl')

def extract_features(X, sample_rate, chroma=True, mfcc=True, mel=True):
    result = np.array([])
    if chroma:
        stft = np.abs(librosa.stft(X))
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result

def convert_audio_to_wav(input_file, output_file):
    try:
        audio_segment = pydub.AudioSegment.from_file(input_file)
        audio_segment.export(output_file, format="wav")
    except Exception as e:
        print("Error converting audio file:", e)



def predict_emotion_with_hugging_face_api(recorded_audio_blob):
    # Prepare the audio file for sending to the Hugging Face API
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
        convert_audio_to_wav(recorded_audio_blob, temp_wav_file.name)

        # Read the audio file and extract features
        with sf.SoundFile(temp_wav_file.name) as sound_file:
            X = sound_file.read(dtype='float32')
            sample_rate = sound_file.samplerate
    
    # Extract features and reshape for prediction
    feature = np.reshape(extract_features(X, sample_rate), (1, -1))

    # Prepare the data to be sent to the Hugging Face API
    files = {'file': open(temp_wav_file.name, 'rb')}

    # Send a POST request to the Hugging Face API endpoint
    response = requests.post('https://api-inference.huggingface.co/models/DunnBC22/wav2vec2-base-Speech_Emotion_Recognition', files=files)


    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        emotion = data['predictions'][0]
        print("Predicted emotion:", emotion)  # Debugging
        return jsonify({'predictions': emotion})
    else:
        print("Error predicting emotion:", response.text)  # Debugging
        return jsonify({'error': 'Failed to predict emotion'})

@app.errorhandler(RequestEntityTooLarge)
def handle_max_file_size_exceeded(error):
    return jsonify({'error': 'File size exceeds the maximum allowed limit (16MB).'}), 413

@app.route('/')
def index():
    return render_template('home.html')
@app.route('/upload.html')
def upload():
    return render_template('upload.html')
@app.route('/record.html')
def record():
    return render_template('record.html')
@app.route('/option.html')
def option():
    return render_template('option.html')
@app.route('/home.html')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' in request.files:
        uploaded_audio = request.files['audio']
        filename = secure_filename(uploaded_audio.filename)
        file_extension = os.path.splitext(filename)[1]

        if file_extension not in ['.wav', '.mp3', '.ogg', '.flac']:
            return jsonify({'error': 'Invalid file format. Please upload a supported audio file (.wav, .mp3, .ogg, .flac).'}), 400

        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, dir=app.config['UPLOAD_FOLDER'], suffix='.wav')
        temp_audio_file_path = temp_audio_file.name

        try:
            uploaded_audio.save(temp_audio_file_path)
        except Exception as e:
            print(f"Error while saving the file: {e}")
            return jsonify({'error': 'Unable to save the uploaded file.'}), 500

        try:
            with sf.SoundFile(temp_audio_file_path) as sound_file:
                X = sound_file.read(dtype='float32')
                sample_rate = sound_file.samplerate
        except Exception as e:
            print(f"Error while reading the audio file: {e}")
            return jsonify({'error': 'Unable to read the uploaded audio file.'}), 500

        try:
            feature = np.reshape(extract_features(X, sample_rate), (1, -1))
            emotion_probabilities = model.predict_proba(feature)
            emotion = np.argmax(emotion_probabilities)
            print("Predicted emotion:", emotion)  # Debugging
            return jsonify({'predictions': emotion.tolist(), 'probabilities': emotion_probabilities.tolist()})
        except Exception as e:
            print(f"Error while predicting emotion: {e}")
            return jsonify({'error': 'Unable to predict emotion from the uploaded audio file.'}), 500

@app.route('/api/recognize', methods=['POST'])
def recognize_emotion():
    audio_file = request.files['audio']
    audio_data = audio_file.read()

    headers = {
        'Authorization': 'Bearer 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'
    }

    response = requests.post('https://api-inference.huggingface.co/models/DunnBC22/wav2vec2-base-Speech_Emotion_Recognition', headers=headers, data=audio_data)
    data = response.json()
    print(data)

    # Get the emotions and scores from the response
    emotions = [d['label'] for d in data]
    scores = [d['score'] for d in data]

    # Sort the emotions and scores based on the score values
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    sorted_emotions = [emotions[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]

    # Create the data list with sorted emotions and scores
    data = [
        {'label': sorted_emotions[i], 'score': sorted_scores[i]}
        for i in range(len(sorted_scores))
    ]

    return jsonify({
        'emotion': data[0]['label'],
        'confidence': data[0]['score'],
        'emotion_scores': [d['score'] for d in data],
        'emotion_labels': sorted_emotions
    })



if __name__ == '__main__':
    app.run(debug=True)
