# 🎙️ Audio Transcription App

A web app that converts speech in audio files to text, built on **OpenAI's Whisper** automatic speech recognition (ASR) model with a **Gradio** interface.

## 📌 Problem
Transcribing meetings, lectures, interviews, and voice notes by hand is slow. This app gives anyone a simple way to upload an audio file and get an accurate transcript in seconds, with no technical setup.

## ⚙️ How it works
1. The user uploads an audio file through the Gradio interface.
2. The Hugging Face `pipeline("automatic-speech-recognition")` loads **`openai/whisper-tiny.en`**, a lightweight English Whisper model.
3. Long recordings are split into **30-second chunks** and processed in batches (`batch_size=8`), so audio of any length can be transcribed.
4. The chunks are joined back together and the full transcript appears in the app.

<!-- Add a screenshot of the app here: ![App screenshot](screenshot.png) -->

## 🚀 How to run
```bash
git clone https://github.com/youstinashraf/Audio-Transcription-App.git
cd Audio-Transcription-App
pip install -r requirements.txt
python speech2text_app.py
```
Then open the local link shown in the terminal (usually http://127.0.0.1:7860) and upload an audio file.

> **Note:** Audio decoding requires [FFmpeg](https://ffmpeg.org/download.html) to be installed on your system.

## 🛠️ Tech stack
Python · OpenAI Whisper · Hugging Face Transformers · PyTorch · Gradio

## 🔮 Future improvements
- Load the model once at startup to speed up every request after the first
- Support more languages (e.g., Arabic) with a multilingual Whisper model
- Add microphone recording and downloadable transcripts (`.txt` / `.srt` subtitles)
- Deploy publicly on Hugging Face Spaces
