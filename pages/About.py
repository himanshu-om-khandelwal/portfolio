import streamlit as st

content = open('content/about.md', 'r').read()

st.markdown(content)
