import streamlit as st

content = open('content/projects.md', 'r').read()

st.markdown(content)