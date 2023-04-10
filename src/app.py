from typing import List, Dict
import os
import io
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_index import GPTSimpleVectorIndex

if os.environ.get('ENV') == 'prod':
    env_file = 'app.env'
else:
    env_file = 'local.env'

with open(env_file, 'r') as f:
    env_vars = f.read().splitlines()

for env_var in env_vars:
    key, value = env_var.split('=')
    os.environ[key] = value

#api_key = os.environ.get('OPENAI_API_KEY')

#os.environ["OPENAI_API_KEY"] = api_key

index = GPTSimpleVectorIndex.load_from_disk('doc-indices/docs_index.json')

app = FastAPI()

# configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    content: str
    role: str

class Model(BaseModel):
    id: str
    maxLength: int
    name: str
    tokenLimit: int

class ChatPayload(BaseModel):
    key: str
    messages: List[Message]
    model: Model
    prompt: str

"""
Model send from client to server
{
    "key": "",
    "messages": [
        {
            "content": "write code to resize image in python",
            "role": "user"
        }
    ],
    "model": {
        "id": "gpt-3.5-turbo",
        "maxLength": 12000,
        "name": "GPT-3.5",
        "tokenLimit": 4000
    },
    "prompt": "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown."
}
"""

@app.get("/hello")
async def hello():    
    return {"message": "Hello from Monday Bot Server!"}

def generate(long_string: str):
        # Create a byte stream from the long string
        stream = io.BytesIO(long_string.encode())

        # Yield the stream data in chunks
        while True:
            data = stream.read(1024)
            if not data:
                break
            yield data

@app.post("/api/bot/ask")
async def ask_bot(question: ChatPayload):
    logging.info(f"Query: {question}")
    if(len(question.messages) == 0):
        return StreamingResponse(generate("Hi, I'm Monday Bot. How can I help you today?"), media_type="text/plain")
    
    query = question.messages[-1].content
    response = index.query(query)
    logging.info(f"Response: {response}")
    return StreamingResponse(generate(response.response), media_type="text/plain")

@app.post("/api/docs/upload")
async def upload_file(file: UploadFile = File(...)):
    with open(f"app_data/{file.filename}", "wb") as buffer:
        buffer.write(await file.read())
    return {"filename": file.filename}
