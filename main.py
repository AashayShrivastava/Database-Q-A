import streamlit as st
from langchain_helper import get_few_shot_db_chain

st.set_page_config(page_title="AtliQ T Shirts Q&A", layout="centered")
st.title("AtliQ T Shirts: Database Q&A 👕")

@st.cache_resource
def load_chain():
    return get_few_shot_db_chain()

chain = load_chain()

question = st.text_input("Question:")

if question:
    with st.spinner("Thinking..."):
        response = chain.run(question)

    st.header("Answer")
    st.write(response)
