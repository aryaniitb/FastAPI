from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

class Query(BaseModel):
    text: str

app = FastAPI()

ml_models = {}

def load_model():
    try:
        model_name = "AKP0032/tiny-llama-colorist-lora2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        ml_models["answer_to_everything"] = (tokenizer, model)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def generate_text(query: Query):
    if "answer_to_everything" not in ml_models:
        raise HTTPException(status_code=500, detail="Model has not been loaded yet.")
    
    tokenizer, model = ml_models["answer_to_everything"]
    inputs = tokenizer(query.text, return_tensors="pt")
    outputs = model.generate(inputs.input_ids)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

@app.post("/generate", response_model=dict)
def generate_text_handler(query: Query):
    try:
        return generate_text(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_model=dict)
def read_root():
    return {"message": "Welcome to the text generation API!"}

# Endpoint to load the model
@app.on_event("startup")
def on_startup():
    load_model()

# Endpoint to clear the model
@app.on_event("shutdown")
def on_shutdown():
    ml_models.clear()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=6099)
