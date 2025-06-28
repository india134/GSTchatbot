from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
import pdfplumber
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate as prompttemplate
from langchain_community.chat_models import ChatOllama




import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "disabled"


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")



app = Flask(__name__)


# Load the document embeddings model
embedding_model= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# load the pdf

loader= PyPDFLoader("gst.pdf")
pdf= loader.load()
# paragraph wise chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(pdf)
llm= ChatOpenAI(
    model_name="gpt-3.5-turbo",  # Specify the model to use
    openai_api_key=openai_api_key
)



template = """
You are an AI assistant helping to answer questions strictly based on the provided context from the GST Act. 
Only answer based on the context given below. 
Do not use any external knowledge, assumptions, or guesses. 
If the information is not present in the context, respond with: 
→ "The information is not available in the provided document."

Context: {context}

Question: {question}

Answer:
"""
prompt= prompttemplate(template=template,
                       input_variables=["context", "question"])

vector_store = Chroma.from_documents(chunks, embedding_model)
# Create a RetrievalQA chain
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,  # Specify the LLM to use
    chain_type_kwargs={"prompt": prompt},  # Optional: set verbose to True for debugging
    chain_type="stuff", 
    retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 7})
)
'''compressor= EmbeddingsFilter(
    embeddings=embedding_model,k=5)'''

@app.route('/')
def home():
    return render_template('bot.html')   # 👈 UI loads here

@app.route('/query', methods=['POST'])
def query():
    question = request.form.get('question', '')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = retrieval_qa.run(question)

    return jsonify({"answer": answer})   # 👈 API returns JSON only





if __name__ == "__main__":
    app.run(debug=True)
