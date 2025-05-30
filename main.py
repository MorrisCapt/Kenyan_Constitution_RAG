# PDF processing
from langchain_community.document_loaders import PyPDFLoader
# Splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Embedding & Chroma DB
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
from langchain.retrievers import MultiQueryRetriever
from langchain.prompts import PromptTemplate
import chromadb
from langchain_chroma import Chroma # Import Chroma class for vectorstore initialization.
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
# Groq for LLM
import groq

import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from jinja2 import Environment, FileSystemLoader

# API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI is working!"}

# Load the pdf file.
loader = PyPDFLoader("constitution_of_kenya.pdf")
pdf = loader.load()

# Initializing the splitters.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pdf)

# Initialize the embedding model.
model = SentenceTransformerEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1", show_progress=True)

# Prepare texts for embedding
texts = [chunk.page_content for chunk in chunks]
metadatas = [chunk.metadata for chunk in chunks]

# Generate embeddings
embeddings_vectors = model.embed_documents(texts)

# Setting up Chroma
client = chromadb.PersistentClient(path = "chroma_store")
collection = client.get_or_create_collection(name = "Kenya_constitution")

# Initialize ChromaDB with updated settings
vectore_store = Chroma(
    collection_name="kenya_constitution",
    embedding_function=model,
    persist_directory="./chroma_store"
)

# Initialize the retriever
question = "What is the structure of the Kenyan government?"
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectore_store.as_retriever(), llm=llm
)

def get_response(question):
    # Retrieve relevant documents
    docs = retriever_from_llm.get_relevant_documents(question)

    # Format the retrieved documents (using a simple prompt template here)
    prompt_template = """You are a constitutional advisor specialized in the Kenyan Constitution.
    Use only the following excerpts from the Kenyan Constitution to answer the question.
    If the specific information isn't found in these excerpts, state that the information
isn't available in the provided constitutional sections rather than speculating.

CONSTITUTIONAL EXCERPTS:
{context}

Question: {question}
Answer based strictly on the Kenyan Constitution:"""


    # Moved the prompt formatting inside the function
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    ).format(context="\n\n".join([doc.page_content for doc in docs]), question=question)

    # Generate a response using the LLM
    response = llm([HumanMessage(content=prompt)])
    print(response)
    return response # Added return statement to return the response

# Example usage
query = input("You:")
response = get_response(query)
print(response)

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# CORS middleware (optional but helpful for testing in browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Jinja2Templates to serve the HTML
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define request body model
class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(q: Question):
    try:
        response = get_response(q.question)
        return {"answer": response.content}  # .content from HumanMessage output
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}
