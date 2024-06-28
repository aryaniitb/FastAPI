from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
import uvicorn
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from langchain_openai import ChatOpenAI

# class Query(BaseModel):
#     prompts: List[str]

app = FastAPI()

llama2 = None
retriever = None
csv_data = None

template = PromptTemplate.from_template("{prompts}")

@app.post("/generate_response")
async def generate_response(query: str):
    try:
        print("invoking...")
        if not llama2 or not retriever or csv_data is None:
            raise HTTPException(status_code=500, detail="System not fully initialized")
        
       
        retrieved_info = retrieve_relevant_info(query)
        
       
        full_prompt = query + " " + " ".join(retrieved_info)
        print(full_prompt)
        messages = [("user", full_prompt)]
        
        # Generate response
        response = llama2.invoke(messages)
        print(response)
        
        return {"response": response}
    
    except Exception as e:
        print("ERROR : ", e)
        raise HTTPException(status_code=500, detail=str(e))

def load_model():
    global llama2
    try:
        llama2 = ChatOpenAI(
        model="llama3:latest",
        openai_api_base="https://923b-35-245-186-65.ngrok-free.app/v1",
        openai_api_key="Not needed for local server")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        llama2 = None

def load_retriever():
    global retriever
    try:
        retriever = SentenceTransformer('all-MiniLM-L6-v2')
        print("Retriever loaded successfully.")
    except Exception as e:
        print(f"Failed to load retriever: {str(e)}")
        retriever = None

def load_csv_data():
    global csv_data
    try:
        # Load the CSV file
        csv_data = pd.read_csv('Housing.csv')
        print("CSV data loaded successfully.")
    except Exception as e:
        print(f"Failed to load CSV data: {str(e)}")
        csv_data = None

def retrieve_relevant_info(query, top_k=3):
    query_embedding = retriever.encode(query, convert_to_tensor=True)
    
    
    csv_data['description'] = csv_data.apply(lambda row: ' '.join(row.astype(str)), axis=1)
    
    csv_embeddings = retriever.encode(csv_data['description'].tolist(), convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, csv_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    retrieved_info = [csv_data['description'].iloc[idx] for idx in top_results.indices.numpy()]
    return retrieved_info

@app.on_event("startup")
async def on_startup():
    load_model()
    load_retriever()
    load_csv_data()

@app.on_event("shutdown")
async def on_shutdown():
    global llama2, retriever, csv_data
    llama2 = None
    retriever = None
    csv_data = None
    print("Resources unloaded.")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9008)
