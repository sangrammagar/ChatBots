import os
from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import InferenceClient

# Load env
load_dotenv()
token = os.getenv("HUGGINGFACE_API_KEY")

client = InferenceClient(api_key=token)

st.title("🤖 AI Chatbot")

# Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
user_input = st.chat_input("Ask something...")

if user_input:
    # store user msg
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            completion = client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                messages=st.session_state.messages,
                max_tokens=200
            )

            response = completion.choices[0].message["content"]
            st.write(response)

    # store response
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
