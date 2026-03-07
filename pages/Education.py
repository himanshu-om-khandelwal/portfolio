import streamlit as st

content = open('content/education.md', 'r').read()

st.markdown(content)