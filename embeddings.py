import warnings
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
import uvicorn
from typing import List
import torch
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
# Suppress specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

app = FastAPI()

llama2 = None

client = MongoClient("mongodb+srv://aryan0711:<PASSWORD>@firstcluster.1sukz3j.mongodb.net/")
db = client["chat_datasets"]
session_col = db["chat_session"]
chat_col = db["chat_history"]
vector_col = db["vector_embeddings"]

@app.post("/start_session")
async def start_session():
    try:
        session = "session has been created"
        role = "user and assistant"
        document = {"temp_session": session, "role": role}
        x = session_col.insert_one(document)
        return {"session_id": str(x.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_response")
async def generate_response(query: str, session_id: str):
    try:
        if not llama2:
            raise HTTPException(status_code=500, detail="System not fully initialized")

        previous_interactions = list(chat_col.find({"session_id": session_id}))
        
        context = retrieve_relevant_info(query)
        messages = [
            ("system", f"You are a helpful assistant that gives the most suitable answer from the book you can use {context} provided"),
            ("user", query)
        ]

        document2 = {"session_id": session_id, "role": "user", "content": query}
        user_doc_id = chat_col.insert_one(document2).inserted_id
        
        response = llama2.invoke(messages)
        
        document3 = {"session_id": session_id, "role": "assistant", "content": response.content, "query_id": user_doc_id}
        chat_col.insert_one(document3)
        
        chat_record = {"user": query, "assistant_response": response.content}
        previous_interactions.append(document2)
        previous_interactions.append(document3)
       
        recent_interactions = previous_interactions[-6:]
        formatted_response = [{"role": doc["role"], "content": doc["content"]} for doc in recent_interactions]

        return {"response": formatted_response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def retrieve_relevant_info(query: str):
    try:
        # Embed the query using the global 'embeddings' model
        embeddings = OllamaEmbeddings(base_url='https://9bed-34-143-156-83.ngrok-free.app', model="llama3")
        query_embedding = embeddings.embed_query(query) 

        stored_docs = list(vector_col.find({}, {"embedding": 1, "text": 1, "_id": 0}))
        # Pad the embeddings to a consistent length
        max_length = max([len(doc["embedding"]) for doc in stored_docs])
        stored_embeddings = np.array([doc["embedding"] + [0] * (max_length - len(doc["embedding"])) for doc in stored_docs])
        stored_texts = [doc["text"] for doc in stored_docs]

        query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
        stored_embeddings = torch.tensor(stored_embeddings, dtype=torch.float32)

        # Assuming 'util.cos_sim' is defined elsewhere
        scores = util.cos_sim(query_embedding, stored_embeddings) 

        top_k_indices = scores.topk(3).indices.tolist()[0]
        top_k_docs = [stored_texts[idx] for idx in top_k_indices]

        return " ".join(top_k_docs)
    except Exception as e:
        raise Exception(f"Error retrieving relevant info: {str(e)}")

def load_model():
    global llama2
    try:
        llama2 = ChatOpenAI(
            model="llama3:latest",
            openai_api_base="https://9bed-34-143-156-83.ngrok-free.app/v1",
            openai_api_key="Not needed for local server"
        )
    except Exception as e:
        llama2 = None
        raise Exception(f"Failed to load model: {str(e)}")

@app.on_event("startup")
async def on_startup():
    try:
        load_model()
    except Exception as e:
        raise Exception(f"Error during startup: {str(e)}")

@app.on_event("shutdown")
async def on_shutdown():
    global llama2
    try:
        llama2 = None
    except Exception as e:
        raise Exception(f"Error during shutdown: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9008)
