import streamlit as st

st.set_page_config(
    page_title="app",
    layout="wide",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    Index page
    """
)
