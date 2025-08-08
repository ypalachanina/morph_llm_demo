from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from utils.streamlit_utils import init_session, display_sidebar, video_mode, camera_mode


def run_app():
    st.set_page_config(page_title="Morpheus LLM PoC", page_icon="👓")
    st.title("Morpheus LLM PoC")
    session = st.session_state
    init_session(session)
    try:
        display_sidebar(session)
        if session["mode"] == 'video':
            video_mode(session)
        elif session["mode"] == 'camera':
            camera_mode(session)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    run_app()
