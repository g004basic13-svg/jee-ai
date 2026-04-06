from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jee_solver import solve

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    text: str

@app.post("/solve")
async def solve_question(q: Question):
    solution, model, similar = solve(q.text)
    return {
        "solution": solution,
        "model": model,
        "similar": similar
    }