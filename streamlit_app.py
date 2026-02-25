import streamlit as st
import uuid
from dotenv import load_dotenv
import requests
load_dotenv(override=True)

# API URL
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_URL = f"{API_BASE_URL}/chat"

# Session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Page config
st.set_page_config(
    page_title="Atlas Horizon Hotel customer assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===============================
# SIDEBAR
# ===============================

with st.sidebar:
    st.title("👤 Guest Info")
    st.markdown("---")

    guest_type = st.selectbox(
        "🏨 Guest Type",
        ["Business", "Leisure", "Family", "Group", "VIP"],
        index=0
    )

    loyalty = st.selectbox(
        "⭐ Loyalty Tier",
        ["Bronze", "Silver", "Gold", "Platinum", "Diamond"],
        index=2
    )

    city = st.text_input("📍 City", "Sydney")

    st.session_state.guest_info = {
        "guest_type": guest_type,
        "loyalty": loyalty,
        "city": city
    }

    st.markdown("---")
    st.write(f"🔐 Session ID: {st.session_state.session_id}")


# ===============================
# CHAT INTERFACE
# ===============================

st.title("🏨 Atlas Horizon Customer Support")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("💬 Type your message..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Call API and get response
    try:
        response = requests.post(
            API_URL,
            json={
                "query": prompt,
                "session_id": st.session_state.session_id,
                "guest_type": st.session_state.guest_info["guest_type"],
                "loyalty": st.session_state.guest_info["loyalty"],
                "city": st.session_state.guest_info["city"]
            }
        )
        response.raise_for_status()
        data = response.json()
        assistant_message = data.get("answer", "Sorry, I couldn't process that. 😔")
    except Exception as e:
        assistant_message = f"⚠️ Error: {str(e)}"

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(assistant_message)