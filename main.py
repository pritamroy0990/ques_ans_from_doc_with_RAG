import streamlit as st
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain


def save_uploaded_file(uploaded_file):
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)  
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path 

st.set_page_config(page_title="Document Q&A", layout="wide") 
st.title("My Document Analysis by deepseek-r1:1.5b")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
ques = st.text_area("Enter your question:", "")

button = st.button("Search", type="primary")

# Cache the vector database and model
@st.cache_resource
def process_document(file_path):
    with st.spinner("Loading and processing the document..."):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    
    with st.spinner("Splitting document into chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        documents = text_splitter.split_documents(docs)
    
    with st.spinner("Creating vector embeddings..."):
        db = FAISS.from_documents(documents, OllamaEmbeddings(model='deepseek-r1:1.5b'))
    
    return db

if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)
    db = process_document(file_path) 

    llm = OllamaLLM(model="deepseek-r1:1.5b")

    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer. 
    <context>
    {context}
    </context>
    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    st.success("Document processed successfully! Ask your question now.")

    if button:
        if ques.strip():
            with st.spinner("AI is thinking..."):
                response = retrieval_chain.invoke({"input": ques})
                answer = response.get("answer", "Sorry, I couldn't find a relevant answer.")

                match = re.search(r"</think>+", answer)
                if match:
                    answer = answer[match.span()[1]:].strip()
                    # answer = answer.strip()

                st.subheader("Answer:")
                st.write(answer)
        else:
            st.warning("Please enter a question before searching.")
