from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
import uvicorn

class Query(BaseModel):
    prompts: List[str]

app = FastAPI()

llama2 = None

template = PromptTemplate.from_template("{prompts}")

@app.post("/generate_response/")
async def generate_response(query: Query):
    try:
        if not llama2:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        
        response = llama2.generate(query.prompts)
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def load_model():
    global llama2
    try:
        llama2 = Ollama(model="llama3", )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        llama2 = None

@app.on_event("startup")
async def on_startup():
    load_model()

@app.on_event("shutdown")
async def on_shutdown():
    global llama2
    llama2 = None
    print("Model unloaded.")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9005)




