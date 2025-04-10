import os
import streamlit as st
from chatbot import extract_all_pdf_text, prepare_vectorstore, create_chatbot

# Set your NVIDIA API Key here
NVIDIA_API_KEY = "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"  # Replace this with your actual key

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄🤖 Chat with Your PDFs (NVIDIA NIM-Powered)")

# Upload PDFs
st.markdown("Upload your PDF files below:")
uploaded_files = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Save uploaded PDFs to a temp folder
    folder_path = "pdfs"
    os.makedirs(folder_path, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(folder_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        st.success(f"✅ Uploaded: {file.name}")

    # Process PDFs and prepare chatbot
    with st.spinner("🔍 Processing PDFs..."):
        text, used_files = extract_all_pdf_text(folder_path)
        if not text.strip():
            st.error("⚠️ No text could be extracted from the uploaded PDFs.")
        else:
            vectorstore = prepare_vectorstore(text)
            chatbot = create_chatbot(vectorstore, NVIDIA_API_KEY)
            st.success("🤖 Chatbot is ready to chat!")

            # Chat interface
            user_input = st.text_input("💬 Ask a question about the PDFs:")
            if user_input:
                with st.spinner("🧠 Thinking..."):
                    response = chatbot.invoke(user_input)
                st.markdown(f"**🤖 Answer:** {response}")
