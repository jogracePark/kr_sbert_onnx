import os
from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime
from transformers import AutoTokenizer
import numpy as np
from typing import List

app = FastAPI()

# Path to the ONNX model and tokenizer files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "kr_sbert_onnx_int8")

# Global variables for tokenizer and ONNX session
tokenizer = None
onnx_session = None
model_output_dim = None # To store the dimension of the embeddings

# Function to perform mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

# Model loading function (executed once at container startup)
def load_model_once():
    global tokenizer, onnx_session, model_output_dim
    if onnx_session is None: # Prevent double loading
        print(f"Loading tokenizer from {MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        print("Tokenizer loaded.")

        print(f"Loading ONNX model from {MODEL_PATH}/model_int8.onnx...")
        onnx_session = onnxruntime.InferenceSession(f"{MODEL_PATH}/model_int8.onnx", providers=['CPUExecutionProvider'])
        print("ONNX model loaded.")
        
        # Determine the output dimension from a dummy inference
        dummy_input = tokenizer(["test"], return_tensors="np", padding='max_length', truncation=True, max_length=128)
        
        print("\n--- Tokenizer dummy_input shapes ---")
        for key, value in dummy_input.items():
            print(f"  {key}: {value.shape}")

        # Filter input_feed to only include inputs expected by the ONNX model
        onnx_input_names = [inp.name for inp in onnx_session.get_inputs()]
        print("\n--- ONNX model expected input shapes ---")
        for inp in onnx_session.get_inputs():
            print(f"  {inp.name}: {inp.shape}")

        input_feed = {}
        if "input_ids" in onnx_input_names:
            input_feed["input_ids"] = dummy_input["input_ids"]
        if "attention_mask" in onnx_input_names:
            input_feed["attention_mask"] = dummy_input["attention_mask"]
        if "token_type_ids" in onnx_input_names:
            input_feed["token_type_ids"] = dummy_input["token_type_ids"]

        dummy_output = onnx_session.run(None, input_feed)
        dummy_embeddings = mean_pooling(dummy_output, dummy_input["attention_mask"])
        model_output_dim = dummy_embeddings.shape[1]
        print(f"Model output dimension: {model_output_dim}")


# App startup event to load the model
load_model_once()

class TextRequest(BaseModel):
    sentences: List[str]

@app.post("/embed")
def embed(req: TextRequest):
    if not tokenizer or not onnx_session:
        raise RuntimeError("Model and tokenizer not loaded. Please check server startup logs.")

    # Tokenize sentences
    encoded_input = tokenizer(
        req.sentences,
        padding='max_length',
        truncation=True,
        max_length=128, # Assuming a max length, adjust if needed
        return_tensors="np"
    )

    # Prepare ONNX input
    onnx_input_names = [inp.name for inp in onnx_session.get_inputs()]
    input_feed = {}
    if "input_ids" in onnx_input_names:
        input_feed["input_ids"] = encoded_input["input_ids"]
    if "attention_mask" in onnx_input_names:
        input_feed["attention_mask"] = encoded_input["attention_mask"]
    if "token_type_ids" in onnx_input_names:
        input_feed["token_type_ids"] = encoded_input["token_type_ids"]

    # Run ONNX inference
    model_output = onnx_session.run(None, input_feed)

    # Perform mean pooling
    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return {
        "dim": model_output_dim,
        "embeddings": embeddings.tolist()
    }

@app.get("/")
def health():
    return {"status": "ok"}