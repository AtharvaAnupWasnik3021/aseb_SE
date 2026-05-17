from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from decision_agent import DecisionAgent


app = FastAPI(title="ProteinAI Decision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    protein_input: str = Field(..., min_length=1)
    gene_symbol: str = Field(..., min_length=1)
    disease_name: str = Field(..., min_length=1)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
def analyze(payload: AnalysisRequest):
    try:
        result = DecisionAgent(
            protein_input=payload.protein_input.strip(),
            gene_symbol=payload.gene_symbol.strip(),
            disease_name=payload.disease_name.strip(),
        ).run()
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
