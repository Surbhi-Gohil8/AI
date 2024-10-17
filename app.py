import streamlit as st
import moviepy.editor as mp
import openai
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech
import os

# Set the path to your Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\surbh\OneDrive\Desktop\New folder (2)\AI\google_credentials.json"

# Azure OpenAI API details
openai.api_key = '22ec84421ec24230a3638d1b51e3a7dc'

# Helper Function 1: Extract Audio from Video
def extract_audio(video_path):
    video = mp.VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')  # Ensure the audio is in WAV format
    return audio_path

# Helper Function 2: Transcribe Audio using Google Speech-to-Text
def transcribe_audio(audio_file_path):
    client = speech.SpeechClient()

    with open(audio_file_path, "rb") as audio_file:
        audio = speech.RecognitionAudio(content=audio_file.read())

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,  # Adjust this to your audio's sample rate
        language_code="en-US",
    )

    try:
        response = client.recognize(config=config, audio=audio)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        return transcript
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

# Helper Function 3: Correct Transcription using GPT-4
def correct_transcription(raw_transcription):
    prompt = f"Please correct the following transcription by removing any grammatical mistakes and filler words like ums, hmms: \n{raw_transcription}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message['content'].strip()

# Helper Function 4: Convert Text to Audio using Google Text-to-Speech
def convert_text_to_audio(corrected_text, output_audio_path):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=corrected_text)

    # Using "Journey" voice model (customizable)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-J",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    with open(output_audio_path, "wb") as out:
        out.write(response.audio_content)
    return output_audio_path

# Helper Function 5: Replace Audio in Video
def replace_audio_in_video(video_path, new_audio_path, output_video_path):
    video = mp.VideoFileClip(video_path)
    new_audio = mp.AudioFileClip(new_audio_path)
    final_video = video.set_audio(new_audio)
    final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

# Streamlit Application
st.title("AI-Powered Audio Replacement")

# Step 1: Upload Video File
uploaded_video = st.file_uploader("Upload a Video File", type=["mp4", "mov", "avi"])

if uploaded_video:
    video_path = f"temp_video.{uploaded_video.name.split('.')[-1]}"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    
    st.video(video_path)
    
    if st.button("Process Video"):
        # Step 2: Extract Audio
        st.write("Extracting audio from video...")
        audio_path = extract_audio(video_path)

        # Step 3: Transcribe Audio
        st.write("Transcribing audio...")
        raw_transcription = transcribe_audio(audio_path)
        
        if raw_transcription:
            st.write(f"Raw Transcription: {raw_transcription}")

            # Step 4: Correct Transcription using GPT-4
            st.write("Correcting transcription using GPT-4...")
            corrected_transcription = correct_transcription(raw_transcription)
            st.write(f"Corrected Transcription: {corrected_transcription}")

            # Step 5: Convert Text to Speech
            st.write("Converting corrected text to speech...")
            output_audio_path = "new_audio.wav"
            convert_text_to_audio(corrected_transcription, output_audio_path)

            # Step 6: Replace Audio in Video
            st.write("Replacing audio in the original video...")
            output_video_path = "final_output_video.mp4"
            replace_audio_in_video(video_path, output_audio_path, output_video_path)

            st.success("Video processed successfully!")
            st.video(output_video_path)

            with open(output_video_path, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
        else:
            st.error("Failed to transcribe audio.")
