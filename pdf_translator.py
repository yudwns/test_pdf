from datetime import datetime
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from pathlib import Path

from style import load_css, load_header  # design .py 추가 추가

# Initialize the OpenAI client
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def extract_main_content_with_gpt(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Updated to gpt-4o-mini
        messages=[
            {"role": "system", "content": """You are an expert in children's literature and story analysis. 
            Your task is to carefully read through a storybook and extract only the main content, 
            focusing on the key elements that make up the story's narrative.

            Please follow these guidelines:
            1. Identify the main characters and their roles in the story.
            2. Summarize the primary plot points and the overall story arc.
            3. Capture any significant themes or morals present in the story.
            4. Note any recurring symbols or motifs that are central to the narrative.
            5. Exclude minor details, repetitive elements, or descriptions that aren't crucial to understanding the core story.
            6. Maintain the essence of the story's language and tone, especially if it's distinctive.
            7. If there are illustrations, briefly mention their significance only if they add crucial information not present in the text.

            Your output should be a concise yet comprehensive representation of the story's main content, 
            suitable for someone who wants to understand the core narrative without reading the entire book."""},
            {"role": "user", "content": f"Please extract the main content from this storybook:\n\n{text}"}
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content

def translate_text_with_gpt(extracted_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Updated to gpt-4o-mini
        messages=[
            {"role": "system", "content": """You are an expert translator specializing in children's literature. 
            Your task is to translate the main content of a storybook from English to Korean. 
            Please ensure that you:
            1. Maintain the story's tone and style in the target language.
            2. Accurately convey the main plot points, characters, and themes.
            3. Adapt any cultural references or idioms appropriately for a Korean audience.
            4. Preserve the essence of any moral lessons or educational content.
            5. Keep the language appropriate for the intended age group of the original story."""},
            {"role": "user", "content": f"Please translate the following extracted main content of a storybook to Korean:\n\n{extracted_text}"}
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content

def speech(content):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    speech_file_path = Path(__file__).parent / f"speech_{timestamp}.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=content
    )
    with open(speech_file_path, 'wb') as file:
        file.write(response.content)
    return speech_file_path

def main():
    # Apply CSS  
    st.markdown(load_css(), unsafe_allow_html=True)  
    
    # Add header  
    st.markdown(load_header(), unsafe_allow_html=True)  


    st.title('Storybook Content Extractor with Audio Conversion (GPT-4o-mini)!!')
    uploaded_file = st.file_uploader("Upload a PDF storybook", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner('Extracting text from PDF...'):
            extracted_text = extract_text_from_pdf(uploaded_file)
            st.text_area("Extracted Raw Text:", extracted_text, height=250)
        
        if st.button('Extract Main Content and Translate'):
            with st.spinner('Extracting main content with GPT-4o-mini...'):
                main_content = extract_main_content_with_gpt(extracted_text)
                st.text_area("Extracted Main Content:", main_content, height=250)
                
                with st.spinner('Translating to Korean with GPT-4o-mini...'):
                    translated_text = translate_text_with_gpt(main_content)
                    st.text_area("Translated Main Content (Korean):", translated_text, height=250)
                
                with st.spinner('Generating audio...'):
                    audio_file = speech(translated_text)
                    audio_bytes = open(audio_file, 'rb').read()
                    st.audio(audio_bytes, format='audio/mp3')

if __name__ == "__main__":
    main()
