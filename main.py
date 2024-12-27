from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from PyPDF2 import PdfReader
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone, ServerlessSpec
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone with new SDK
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

INDEX_NAME = "pdf-chat-index"

# Create Pinecone index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # OpenAI embeddings dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Choose appropriate region
        )
    )

# Get the index instance
index = pc.Index(INDEX_NAME)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversation chains for different sessions
conversation_chains = {}
chat_histories = {}


class ChatRequest(BaseModel):
    session_id: str
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    answer: str
    chat_history: List[Dict[str, str]]


def get_pdf_text(pdf_file_bytes: bytes) -> str:
    """Extract text from PDF bytes"""
    pdf_file = io.BytesIO(pdf_file_bytes)
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text: str) -> List[str]:
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks: List[str], session_id: str):
    """Create vector store from text chunks using Pinecone"""
    embeddings = OpenAIEmbeddings()

    # Create namespace for this session
    namespace = f"session_{session_id}"

    # Initialize Pinecone vector store
    vectorstore = PineconeVectorStore.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=namespace
    )
    print(vectorstore)
    return vectorstore


def get_conversation_chain(vectorstore, session_id: str):
    """Create conversation chain"""
    llm = ChatOpenAI(temperature=0.7)

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), session_id: str = None):
    """Upload and process a PDF file"""
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")

    try:
        contents = await file.read()

        # Process PDF
        if not contents:
            raise HTTPException(status_code=400, detail="Empty PDF file")

        raw_text = get_pdf_text(contents)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="PDF has no extractable text")

        text_chunks = get_text_chunks(raw_text)
        print(text_chunks)

        # Delete and create vectors
        # index.delete(namespace=f"session_{session_id}")
        print("")
        vectorstore = get_vectorstore(text_chunks, session_id)
        print(vectorstore,"")
        # Store conversation chain
        conversation_chains[session_id] = get_conversation_chain(vectorstore, session_id)
        chat_histories[session_id] = []

        return {"message": "PDF processed successfully"}
    except HTTPException as e:
        raise e  # Bubble up HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/chat/", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """Handle chat interaction"""
    if chat_request.session_id not in conversation_chains:
        raise HTTPException(status_code=400, detail="No processed document found for this session")

    try:
        # Update chat history from frontend if provided
        if chat_request.chat_history:
            chat_histories[chat_request.session_id] = chat_request.chat_history

        # Get response from conversation chain
        conversation = conversation_chains[chat_request.session_id]
        response = conversation({'question': chat_request.question})

        # Format chat history
        chat_history = []
        for i, message in enumerate(response['chat_history']):
            role = "user" if i % 2 == 0 else "assistant"
            chat_history.append({
                "role": role,
                "content": message.content
            })

        # Store updated chat history
        chat_histories[chat_request.session_id] = chat_history

        return ChatResponse(
            answer=response['answer'],
            chat_history=chat_history
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    """Retrieve chat history for a session"""
    if session_id not in chat_histories:
        raise HTTPException(status_code=404, detail="Chat history not found")

    return {"chat_history": chat_histories[session_id]}


@app.delete("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up session data and vectors"""
    try:
        # Delete vectors from Pinecone
        index.delete(namespace=f"session_{session_id}")

        # Clear conversation chain and chat history
        if session_id in conversation_chains:
            del conversation_chains[session_id]
        if session_id in chat_histories:
            del chat_histories[session_id]

        return {"message": "Session cleaned up successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)