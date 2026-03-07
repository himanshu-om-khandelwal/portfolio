import streamlit as st

content = open('content/experience.md', 'r').read()

st.markdown(content)