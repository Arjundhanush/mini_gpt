
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------------
# 1. Locate the model directory
# --------------------------------

MODEL_DIR = Path(__file__).resolve().parent / "model"

sys.path.insert(0, str(MODEL_DIR))

from data import tokenizer
from gpt import GPTLanguageModel


# --------------------------------
# 2. FastAPI configuration
# --------------------------------

app = FastAPI(title="Mini-GPT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):
    message: str


# --------------------------------
# 3. Load trained GPT model
# --------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=32,
    block_size=8,
    num_heads=4,
    num_layers=2
).to(device)

checkpoint_path = MODEL_DIR / "mini_gpt.pth"

model.load_state_dict(
    torch.load(checkpoint_path, map_location=device)
)

model.eval()

print("Mini-GPT model loaded successfully!")


# --------------------------------
# 4. Home endpoint
# --------------------------------

@app.get("/")
def home():
    return {"message": "Welcome to Mini-GPT API"}


# --------------------------------
# 5. Chat endpoint
# --------------------------------

@app.post("/chat")
@torch.no_grad()
def chat(request: ChatRequest):

    user_message = request.message.strip().lower()

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    # Check whether every character exists in vocabulary
    unsupported = sorted(set(user_message) - set(tokenizer.stoi))

    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported characters: {unsupported}"
        )

    # Encode user message
    encoded = tokenizer.encode(user_message)

    context = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device
    )

    # Generate new characters
    generated = model.generate(
        context,
        max_new_tokens=100,
        temperature=0.8
    )

    # Extract only newly generated characters
    new_tokens = generated[0, len(encoded):].tolist()

    response_text = tokenizer.decode(new_tokens)

    return {
        "user_message": user_message,
        "response": response_text
    }
