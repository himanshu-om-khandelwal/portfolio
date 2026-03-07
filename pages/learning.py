import streamlit as st

content = open('content/learning.md').read()

st.markdown(content)