from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app= FastAPI(title="Mini-GPT API");

#allow react front end to communicate with FastAPI

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

#Request structure
class chatRequest(BaseModel):
    message: str

#home endpoint
@app.get("/")
def home():
    return {"message": "Welcome to Mini-GPT API"}

#chat endpoint
@app.post("/chat")
def chat(request: chatRequest):
    user_message = request.message

    #temporary response for testing
    return{
        "user_message": user_message,
        "response": "This is a temporary response from Mini-GPT."
    }