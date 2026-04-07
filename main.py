from fastapi import FastAPI
from pydantic import BaseModel
import torch
import tiktoken
from model import GPT
from fastapi.middleware.cors import CORSMiddleware

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_fineweb   = torch.load("model_weights/ckpt_0023000_fineweb_edu.pt", map_location=device, weights_only=False)
ckpt_tiny_stories   = torch.load("model_weights/ckpt_0015500_tiny_stories.pt", map_location=device, weights_only=False)

fineweb_model  = GPT().to(device)
fineweb_model.load_state_dict(ckpt_fineweb["model"])
fineweb_model.eval()

tiny_stories_model = GPT().to(device)
tiny_stories_model.load_state_dict(ckpt_tiny_stories["model"])
tiny_stories_model.eval()

enc = tiktoken.get_encoding("gpt2")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt:      str
    max_tokens:  int   = 30
    temperature: float = 0.8
    top_k:       int   = 14

class GenerateResponse(BaseModel):
    generated: str

@app.post("/fineweb-edu/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    tokens = enc.encode(req.prompt)
    idx    = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        out = fineweb_model.generate(idx, req.max_tokens,
                             temperature=req.temperature,
                             top_k=req.top_k)
    text = enc.decode(out[0].tolist())
    return GenerateResponse(generated=text)

@app.post("/tiny-stories/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    tokens = enc.encode(req.prompt)
    idx    = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        out = tiny_stories_model.generate(idx, req.max_tokens,
                             temperature=req.temperature,
                             top_k=req.top_k)
    text = enc.decode(out[0].tolist())
    return GenerateResponse(generated=text)

@app.get("/test")
def health():
    return {"status": "ok"}