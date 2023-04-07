import streamlit as st
import os

from components.faq import faq

def set_openai_api_key(api_key: str):
    st.session_state["OPENAI_API_KEY"] = api_key

def my_poorly_documented_function():
    st.caption('help')

def sidebar():
    with st.sidebar:
        faq()
        
        st.markdown(
            "#### How to use\n"
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below 🔑\n"  # noqa: E501
            "2. Upload a pdf, docx, or txt file 📄\n"
            "3. Interact with document GPT 💬\n"
        )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
            #value=st.session_state.get("OPENAI_API_KEY", ""),
            value=os.environ["OPENAI_API_KEY"],
        )

        if api_key_input:
            set_openai_api_key(api_key_input)

        