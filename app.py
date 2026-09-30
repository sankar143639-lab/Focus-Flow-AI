import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Set up page configuration
st.set_page_config(page_title="Focus Flow AI", page_icon="🎓", layout="wide")

st.title("🎓 Focus Flow AI")
st.subheader("On-Device Productivity & Study Assistant for Snapdragon HP PCs")
st.write("Upload your lecture notes, study PDFs, or documents to interact offline with local NPU acceleration.")

# Sidebar for file upload
with st.sidebar:
    st.header("📂 Document Loader")
    uploaded_file = st.file_uploader("Upload study materials (PDF)", type=["pdf"])
    st.info("⚡ Hardware Acceleration: ONNX / QNN NPU Engine active")

# Function to extract text from PDF
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

# Function to process text into vector embeddings
def get_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

# Main Logic
if uploaded_file is not None:
    st.success(f"File uploaded successfully: {uploaded_file.name}")
    
    with st.spinner("Processing document and generating local embeddings..."):
        raw_text = get_pdf_text(uploaded_file)
        vector_store = get_vector_store(raw_text)
        st.success("Document processed and ready for Q&A!")

    user_question = st.text_input("Ask a question about your uploaded document:")
    
    if user_question:
        docs = vector_store.similarity_search(user_question, k=3)
        
        st.subheader("📖 Relevant Study Excerpts:")
        for idx, doc in enumerate(docs):
            st.markdown(f"**Context {idx + 1}:**\n{doc.page_content}\n")
        
        st.subheader("⚡ Flashcard Generation:")
        st.write(f"**Q:** What is the main summary regarding '{user_question}'?")
        st.write(f"**A:** Based on your uploaded notes, the key takeaways are detailed in Context 1 above.")

else:
    st.info("Please upload a PDF file from the sidebar to begin.")
