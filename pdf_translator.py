from datetime import datetime
import streamlit as st
import openai
from PyPDF2 import PdfReader
from pathlib import Path
import requests

# Set the OpenAI API key
openai.api_key = st.secrets["openai"]["api_key"]

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def summarize_with_gpt(text):
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Please summarize the following text:\n{text}"}
            ],
            max_tokens=2000
        )
        return response.choices[0].message.content

def translate_text_with_gpt(summarized_text):
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"번역해줘:\n{summarized_text}"}
            ],
            max_tokens=2000
        )
        return response.choices[0].message.content

def speech(content):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    speech_file_path = Path(__file__).parent / f"speech_{timestamp}.mp3"

    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=content
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path

def main():
    st.title('PDF Summarizer with Audio Conversion')

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner('Extracting text...'):
            extracted_text = extract_text_from_pdf(uploaded_file)
            st.text_area("Extracted Text:", extracted_text, height=250)

        if st.button('Summarize'):
            with st.spinner('Summarizing...'):
                summarized_text = summarize_with_gpt(extracted_text)
                st.text_area("Summarized Text:", summarized_text, height=150)

                target_language = "Korean"
                with st.spinner('Translating...'):
                    translated_text = translate_text_with_gpt(summarized_text)
                    st.text_area("Translated Text (Korean):", translated_text, height=150)

                with st.spinner('Generating audio...'):
                    audio_file = speech(translated_text)
                    audio_bytes = open(audio_file, 'rb').read()
                    st.audio(audio_bytes, format='audio/mp3')

if __name__ == "__main__":
    main()
