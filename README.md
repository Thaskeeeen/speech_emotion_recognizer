# **Speech Emotion Recognition (SER) System**

## **Overview**

This project focuses on developing a **Speech Emotion Recognition (SER)** system capable of identifying and classifying various emotional states conveyed through speech signals. The system differentiates between emotions such as **happiness, sadness, anger, fear,** and **neutrality** based on acoustic features extracted from audio recordings.

## **Features Extracted**

The following features are extracted from the audio files:

- **Mel-Frequency Cepstral Coefficients (MFCC):** Captures the power spectrum of a sound, which is critical for distinguishing between different emotions.
- **Chroma Features:** Represents the intensity of the 12 different pitch classes, capturing the harmonic characteristics of the audio.
- **Mel Spectrogram:** Provides a visual representation of the spectrum of frequencies in a sound signal as it varies with time.

## **Dataset**

The dataset used in this project consists of audio recordings with the following characteristics:

- **Speaker:** Single female speaker with an Indian English accent.
- **Emotional States:** Neutral, angry, calm, fear, happy, and sad.
- **Recording Details:** Each audio file is recorded at a frequency of **22.05 kHz** and has a duration of **3-5 seconds**.

The dataset is sourced from a GitHub repository: [Emotion-TTS Dataset](https://github.com/skit-ai/emotion-tts-dataset).


## **Model Training and Evaluation**
The Multi-Layer Perceptron (MLP) Classifier is used to classify the emotions. The model is trained using the features extracted from the audio files, and the performance is evaluated using metrics such as accuracy and classification report.

Results
After training, the model achieved an accuracy of 94% on the test dataset. The following is a detailed classification report:




#### **Acknowledgments**

Special thanks to the creators of the Emotion-TTS Dataset for providing the audio data used in this project.
