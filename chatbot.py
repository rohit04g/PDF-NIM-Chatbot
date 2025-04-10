import os
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# === 1. Extract Text from PDFs ===
def extract_all_pdf_text(folder_path):
    text = ""
    used_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            try:
                doc = fitz.open(filepath)
                for page in doc:
                    text += page.get_text()
                doc.close()
                used_files.append(filename)
                print(f"✅ Processed: {filename}")
            except Exception as e:
                print(f"❌ Failed to read {filename}: {e}")
    return text, used_files


# === 2. Create Vector Store ===
def prepare_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"🧠 Total Chunks Created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# === 3. Set Up NVIDIA NIM Chatbot ===
def create_chatbot(vectorstore, api_key):
    llm = ChatOpenAI(
        model="mistralai/mistral-small-24b-instruct",
        openai_api_key=api_key,
        openai_api_base="https://integrate.api.nvidia.com/v1",
        temperature=0.3
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

# === 4. Run Main Program ===
def main():
    folder_path = "pdfs"
    text, used_files = extract_all_pdf_text(folder_path)
    if not text.strip():
        print("⚠️ No text found in PDFs. Exiting.")
        return

    print("\n📂 Files used:")
    for f in used_files:
        print(" -", f)

    vectorstore = prepare_vectorstore(text)

    NVIDIA_API_KEY = "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"  # 🔐 Replace with your real key
    chatbot = create_chatbot(vectorstore, NVIDIA_API_KEY)

    print("\n🤖 Chatbot is ready. Ask questions! (Type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break
        try:
            response = chatbot.invoke(user_input)
            print("Bot:", response)
        except Exception as e:
            print("⚠️ Error:", e)

if __name__ == "__main__":
    main()
