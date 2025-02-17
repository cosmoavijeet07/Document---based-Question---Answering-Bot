import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai
from PyPDF2 import PdfReader
from huggingface_hub import login
import cassio
import os
import time
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
from pydantic import Field, BaseModel
# import gdown

class GeminiLLM(LLM, BaseModel):
    model_name: str = Field(default="gemini-1.5-flash", description="The name of the Gemini model")
    model: Optional[Any] = Field(None, description="The GenerativeModel instance")

    def __init__(self, model_name: str, **data):
        super().__init__(model_name=model_name, **data)
        self.model = genai.GenerativeModel(model_name=self.model_name)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}
    
load_dotenv(Path(".env"))

st.set_page_config(page_title="Chat with Docs", layout="wide", )
st.title("Chat with Docs!!")



pdfreader = PdfReader("A STUDENT GUIDE.pdf")


if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "faiss_vector_index" not in st.session_state:
    st.session_state.faiss_vector_index = None
    
# Upload document function
def process_uploaded_pdf(uploaded_file):
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        raw_text = ''
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        login(token="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")

        embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        faiss_vector_store = FAISS.from_texts([raw_text], embedding_function)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
        )
        texts = text_splitter.split_text(raw_text)
        faiss_vector_store.add_texts(texts[:50])

        st.session_state.faiss_vector_index = VectorStoreIndexWrapper(vectorstore=faiss_vector_store)
        st.session_state.pdf_processed = True
        st.success("PDF processed and database initialized!")
    else:
        st.error("Failed to process the uploaded file.")


# with st.spinner("Loading"):
#     if pdfreader:
#         raw_text = ''
#         for page in pdfreader.pages:
#             content = page.extract_text()
#             if content:
#                 raw_text += content
        
        # if not st.session_state.pdf_processed:
        #     genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        #     login(token="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")
        #     llm=genai.GenerativeModel(model_name='gemini-1.5-flash')

        #     embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        #     faiss_vector_store = FAISS.from_texts([raw_text], embedding_function)
        #     # result = faiss_vector_store.similarity_search("who is Shrirang",k=3)
        #     # print("Result1:", result[2].page_content)

        #     text_splitter = RecursiveCharacterTextSplitter(
        #         chunk_size=800,
        #         chunk_overlap=200,
        #     )

        #     texts = text_splitter.split_text(raw_text)
        #     faiss_vector_store.add_texts(texts[:50])

        #     st.session_state.faiss_vector_index = VectorStoreIndexWrapper(vectorstore=faiss_vector_store)
        #     st.session_state.pdf_processed =True
# Sidebar for uploading documents
st.sidebar.markdown("## **Upload a Document**")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file:
    process_uploaded_pdf(uploaded_file)

st.sidebar.markdown("## **Welcome to Docs the Document Q & A Chatbot**")
st.sidebar.markdown('##### This chatbot is specifically build for EY Techathon 5.0 as part of Q & A Chatbot')
st.sidebar.markdown(' If anything goes wrong do hard refresh by using **Shift** + **F5** key')

def typing_animation(text, speed):
            for char in text:
                yield char
                time.sleep(speed)

if "intro_displayed" not in st.session_state:
    st.session_state.intro_displayed = True
    intro = "Hello, I am Docs, a  Document Q & A Bot"
    intro2= "Chat with Docs"
    st.write_stream(typing_animation(intro,0.02))
    st.write_stream(typing_animation(intro2,0.02))

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

#initialised prePrompt_selected
if "prePrompt_selected" not in st.session_state:
    st.session_state.prePrompt_selected = False

if "btn_selected" not in st.session_state:
    st.session_state.btn_selected = True

#defined callback fn
def btn_callback():
    st.session_state.prePrompted_selected = False
    st.session_state.btn_selected=False

prePrompt = None


prompt = st.chat_input("Chat with Docs..")
 
if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    query_text = prompt.strip()
    gemini_llm = GeminiLLM(model_name='gemini-1.5-flash')
    if st.session_state.faiss_vector_index is not None:
        
        answer = st.session_state.faiss_vector_index.query(query_text, llm=gemini_llm).strip()
        
        typing_speed = 0.02
        if "context" or "no" in answer:
            with st.chat_message("assistant"):
                st.write_stream(typing_animation(answer, typing_speed))
        else:
            with st.chat_message("assistant"):
                st.write_stream(typing_animation(answer,typing_speed))
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("Database not initialized. Kindly reload and upload the PDF first.")
