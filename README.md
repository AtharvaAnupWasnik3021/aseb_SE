# ASEB Drug Target Decision Pipeline

This project evaluates whether a protein target should be considered for drug discovery.

The pipeline uses four independent agents:

- Structure agent: detects binding pockets and estimates structural druggability.
- Biology agent: checks gene-disease evidence from OpenTargets.
- Safety agent: estimates tissue, off-target, and adverse-risk signals.
- Chemistry agent: checks ChEMBL activity and ligand feasibility.

The `DecisionAgent` combines the four outputs into an integrated score, summary, final verdict, and recommendation.

## Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## Run

```powershell
.\.venv\Scripts\python.exe decision_agent.py
```

By default, the sample run evaluates:

- Protein/PDB: `2ITX`
- Gene: `EGFR`
- Disease: `breast cancer`

Edit the values at the bottom of `decision_agent.py` to evaluate another target.
