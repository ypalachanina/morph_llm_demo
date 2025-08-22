from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from utils.streamlit_utils import StreamlitUI


def run_app():
    # host = "local"
    host = "streamlit"
    st.set_page_config(page_title="Morpheus LLM PoC", page_icon="👓")
    st.title("Morpheus LLM PoC")
    session = st.session_state
    ui = StreamlitUI(session, host)
    try:
        ui.display_sidebar()
        ui.start_app()
    except Exception as e:
        st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    run_app()
