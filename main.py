import streamlit as st
from dotenv import load_dotenv
import os
from pdf_loader import load_and_store_pdf, get_vectorstore
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="PDF Chatbot - RAG", layout="centered")

st.title("📄 AI PDF Assistant with Groq & LangChain")
st.markdown("Upload a PDF and ask questions from it using RAG with LLM from Groq.")

# Upload PDF
uploaded_pdf = st.file_uploader("📤 Upload your PDF here", type=["pdf"])

# Initialize vectorstore
vectorstore = None
if uploaded_pdf:
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_pdf.read())
    st.success("✅ PDF uploaded successfully!")
    vectorstore = load_and_store_pdf("temp_uploaded.pdf")

# Ask a question
user_question = st.text_input("❓ Ask a question from your PDF")

# Single 'Get Answer' button
submit = st.button("💬 Get Answer")

if submit:
    if not uploaded_pdf:
        st.warning("📂 Please upload a PDF first.")
    elif not user_question.strip():
        st.warning("✏️ Please enter a question.")
    else:
        with st.spinner("🤖 Generating answer..."):
            llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
            answer = qa_chain.run(user_question)
            st.success("💡 Answer:")
            st.write(answer)
