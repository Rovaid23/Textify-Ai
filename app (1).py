import whisper
import streamlit as st
import tempfile
import os
from io import BytesIO
from pydub import AudioSegment
from gtts import gTTS  # Import Google Text-to-Speech

# Load Whisper model
model = whisper.load_model("base")

st.title("🎙️ Speech-to-Text & Back to Speech (Whisper AI + gTTS)")
st.write("Upload an audio file to transcribe and convert back to speech!")

# File uploader
audio_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    # Convert audio to WAV if needed
    file_extension = audio_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    if file_extension != "wav":
        audio = AudioSegment.from_file(temp_audio_path, format=file_extension)
        temp_audio_path_wav = temp_audio_path.replace(file_extension, "wav")
        audio.export(temp_audio_path_wav, format="wav")
        os.remove(temp_audio_path)  # Remove original temp file
        temp_audio_path = temp_audio_path_wav

    # Audio playback
    st.audio(audio_file, format="audio/wav")

    with st.spinner("Transcribing..."):
        result = model.transcribe(temp_audio_path)
        transcription = result["text"]
        st.success("Transcription Complete!")
        st.write(transcription)

    # Word count
    word_count = len(transcription.split())
    st.write(f"📝 Word Count: **{word_count}** words")

    # Download button for transcription
    output_txt = BytesIO()
    output_txt.write(transcription.encode())
    output_txt.seek(0)
    st.download_button("📥 Download Transcription", output_txt, file_name="transcription.txt", mime="text/plain")

    # Convert transcribed text back to speech using gTTS
    tts = gTTS(text=transcription, lang="en")
    speech_output_path = "output_speech.mp3"
    tts.save(speech_output_path)

    st.success("🔊 Text-to-Speech Conversion Complete!")
    st.audio(speech_output_path, format="audio/mp3")

    # Cleanup temp files
    os.remove(temp_audio_path)
