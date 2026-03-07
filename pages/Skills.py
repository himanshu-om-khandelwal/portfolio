import streamlit as st

content = open('content/skills.md', 'r').read()
st.markdown(content)