import streamlit as st
import requests
import base64
import os
from dotenv import load_dotenv


load_dotenv()
tts_key=os.getenv('TTS_KEY')
# Set page config
st.set_page_config(page_title="EduMate", layout="wide")

# Custom CSS for improved aesthetics
st.markdown("""
<style>
    body {
        background-color: #f0f4f8;
        color: #1e1e1e;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #1e1e1e;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #ffffff;
        color: #1e1e1e;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
        color: #fff;
    }
    .chat-message.bot .message {
        color: #1e1e1e;
    }
</style>
""", unsafe_allow_html=True)

# Function to play base64-encoded audio directly in the frontend
def autoplay_audio_from_base64(audio_base64: str, audio_format="audio/mp3"):
    try: 
        md = f"""
            <audio controls autoplay="true">
            <source src="data:{audio_format};base64,{audio_base64}" type="{audio_format}">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.error(e)

# Function to fetch base64-encoded audio from backend
def fetch_tts_audio(text="Hello"):
    try:
        url = "https://api.sarvam.ai/text-to-speech"

        payload = {
    "inputs": [text],
    "target_language_code": "hi-IN",
    "speaker": "meera",
    "pitch": 0,
    "pace": 1.65,
    "loudness": 1.5,
    "speech_sample_rate": 8000,
    "enable_preprocessing": True,
    "model": "bulbul:v1"
        }
        headers = {"Content-Type": "application/json",'API-Subscription-Key': tts_key}

        response = requests.request("POST", url, json=payload, headers=headers)
        base64=response.text[12:-3]
        return base64
    except requests.RequestException as e:
        st.error(f"Error calling TTS API: {str(e)}")
    return None, 

# Sidebar for selecting mode
st.sidebar.title("Mode Selection")
mode = st.sidebar.radio("Choose a mode:", ("RAG", "Agent"))

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Title
st.title(f"EduMate - a Study Copilot : {mode} Mode")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


full_response = "Hello"
# Chat input
if prompt := st.chat_input(f"Ask me anything... {mode} mode"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
       
        
        # Show a spinner while waiting for the response
        with st.spinner("Thinking..."):
            try:
                # Make a POST request to the appropriate FastAPI backend endpoint
                endpoint = "rag" if mode == "RAG" else "agent"
                response = requests.post(f"http://localhost:8000/{endpoint}", 
                                         json={"question": prompt})
                
                if response.status_code == 200:
                    full_response = response.json()["response"]
                else:
                    full_response = f"I'm sorry, I encountered an error. Status code: {response.status_code}"
            except requests.RequestException as e:
                full_response = f"I'm sorry, I couldn't connect to the server. Error: {str(e)}"

        message_placeholder.markdown(full_response)
        
        
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Fetch and play TTS audio for the bot's response'
if st.button("Play Response Audio"):
    audio_base64 = fetch_tts_audio(st.session_state.messages[-1]["content"])
    audio_format="audio/mp3"
    if audio_base64:
        autoplay_audio_from_base64(audio_base64, audio_format)

# Footer
st.markdown("---")
st.markdown(f"MODE: {'RAG' if mode == 'RAG' else 'Agent'}")
