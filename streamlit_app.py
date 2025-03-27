
    
import streamlit as st
import torch
from transformers import pipeline
import time
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)


# Load Whisper model
@st.cache_resource
def load_model():
    device = "cpu"
    st.warning("Using CPU for transcription. This may be slower.")
    return pipeline("automatic-speech-recognition", "openai/whisper-large-v3", torch_dtype=torch.float16 if device == "cuda" else torch.float32, device=device)

whisper = load_model()

# Streamlit App
st.title("🎙 Audio Transcription with Whisper")
st.write("Upload an audio file to get the transcription in Hinglish.")

# File Upload
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    st.write("Transcribing... Please wait.")

    # Save the file temporarily
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.read())

    # Record start time
    start_time = time.time()
    start_local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))

    # Perform transcription with timestamps for long-form audio
    try:
        transcription = whisper("temp_audio.mp3", return_timestamps=True)
        end_time = time.time()
        end_local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
        total_time = (end_time - start_time)/60

        st.success("Transcription completed!")
        st.write(f"Transcription started at: {start_local_time}")
        st.write(f"Transcription ended at: {end_local_time}")
        st.write(f"Total processing time: {total_time:.2f} Minutes")

        for segment in transcription['chunks']:
            hinglish_text = transliterate(segment['text'], sanscript.ITRANS, sanscript.DEVANAGARI)
            st.write(f"[{segment['timestamp'][0]}s - {segment['timestamp'][1]}s]: {segment['text']}")
    except Exception as e:
        st.error(f"Error: {e}")

    # Clean up
    import os
    os.remove("temp_audio.mp3")
