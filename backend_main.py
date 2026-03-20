import os
import json
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import requests

from Bio.PDB import PDBParser, PDBList
from scipy.spatial import KDTree, Delaunay
from sklearn.cluster import DBSCAN


HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}

app = FastAPI(title="StructureAgent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Core analysis logic (preserved from original)
# ─────────────────────────────────────────

def download_pdb(pdb_id: str, work_dir: str) -> str:
    pdbl = PDBList()
    file_path = pdbl.retrieve_pdb_file(pdb_id, pdir=work_dir, file_format="pdb")
    new_name = os.path.join(work_dir, f"{pdb_id}.pdb")
    if not os.path.exists(new_name) and os.path.exists(file_path):
        os.rename(file_path, new_name)
    return new_name if os.path.exists(new_name) else file_path


def download_alphafold(protein_name: str, work_dir: str) -> Optional[str]:
    try:
        url = f"https://rest.uniprot.org/uniprotkb/search?query={protein_name}&format=json&size=1"
        r = requests.get(url, timeout=15)
        data = r.json()
        if "results" not in data or not data["results"]:
            return None
        uniprot_id = data["results"][0]["primaryAccession"]
        for version in ["v4", "v3"]:
            af_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_{version}.pdb"
            r = requests.get(af_url, timeout=15)
            if r.status_code == 200:
                file_name = os.path.join(work_dir, f"{uniprot_id}.pdb")
                with open(file_name, "wb") as f:
                    f.write(r.content)
                return file_name
        return None
    except Exception:
        return None


def resolve_input(protein_input: str, work_dir: str) -> str:
    clean = protein_input.strip().upper()
    if len(clean) == 4 and clean.isalnum():
        return download_pdb(clean, work_dir)
    query = {
        "query": {
            "type": "group", "logical_operator": "or",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "struct.title", "operator": "contains_words", "value": protein_input}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_polymer_entity.pdbx_description",
                    "operator": "contains_words", "value": protein_input}}
            ]
        },
        "return_type": "entry",
        "request_options": {"pager": {"start": 0, "rows": 1}}
    }
    try:
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=15)
        data = r.json()
        if "result_set" in data and data["result_set"]:
            pdb_id = data["result_set"][0]["identifier"]
            return download_pdb(pdb_id, work_dir)
    except Exception:
        pass
    af = download_alphafold(protein_input, work_dir)
    if af:
        return af
    raise ValueError(f"Structure not found for '{protein_input}'")


def load_structure(pdb_file: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    models = []
    for model in structure:
        atoms, residues, res_ids, chain_ids = [], [], [], []
        for chain in model:
            for residue in chain:
                if residue.get_resname() == "HOH":
                    continue
                res_id = residue.get_id()[1]
                for atom in residue:
                    if atom.get_id() != "CA":
                        continue
                    atoms.append(atom.coord)
                    residues.append(residue.get_resname())
                    res_ids.append(res_id)
                    chain_ids.append(chain.get_id())
        models.append({
            "atoms": np.array(atoms),
            "residues": residues,
            "res_ids": res_ids,
            "chain_ids": chain_ids
        })
    return models


def detect_pocket(atoms: np.ndarray):
    if len(atoms) < 5:
        return None, []
    tri = Delaunay(atoms)
    cavities = []
    cavity_sizes = []
    for simplex in tri.simplices:
        pts = atoms[simplex]
        center = np.mean(pts, axis=0)
        dist = np.min(np.linalg.norm(atoms - center, axis=1))
        if dist > 4:
            cavities.append(center)
            cavity_sizes.append(dist)
    if not cavities:
        return None, []
    cavities = np.array(cavities)
    clustering = DBSCAN(eps=3, min_samples=6).fit(cavities)
    labels = clustering.labels_
    best, size = None, 0
    for label in set(labels):
        if label == -1:
            continue
        cluster = cavities[labels == label]
        if len(cluster) > size:
            best = cluster
            size = len(cluster)
    all_clusters = []
    for label in set(labels):
        if label == -1:
            continue
        cluster = cavities[labels == label]
        all_clusters.append({
            "label": int(label),
            "size": len(cluster),
            "center": np.mean(cluster, axis=0).tolist()
        })
    return best, all_clusters


def extract_pocket_residues(atoms, res_ids, pocket):
    tree = KDTree(atoms)
    pocket_res = set()
    for point in pocket:
        neighbors = tree.query_ball_point(point, r=6)
        for idx in neighbors:
            pocket_res.add(res_ids[idx])
    return sorted(pocket_res)


def hydrophobic_score(residues, pocket_res_ids, res_ids):
    pocket_res = [residues[i] for i, r in enumerate(res_ids) if r in pocket_res_ids]
    if not pocket_res:
        return 0.0
    return sum(1 for r in pocket_res if r in HYDROPHOBIC) / len(pocket_res)


def compactness(pocket):
    center = np.mean(pocket, axis=0)
    dist = np.linalg.norm(pocket - center, axis=1)
    return float(np.std(dist))


def drug_score(volume, depth, hydro):
    score = 0.4 * (volume / 500) + 0.3 * hydro + 0.3 * (depth / 15)
    return round(min(score, 1.0), 3)


def classify_decision(score: float) -> str:
    if score > 0.6:
        return "Likely Druggable"
    elif score > 0.4:
        return "Moderate Potential"
    else:
        return "Poor Drug Target"


def run_analysis(pdb_file: str) -> dict:
    models = load_structure(pdb_file)
    if not models:
        raise ValueError("No models found in structure")
    model = models[0]
    atoms = model["atoms"]
    residues = model["residues"]
    res_ids = model["res_ids"]

    pocket, all_clusters = detect_pocket(atoms)
    if pocket is None:
        raise ValueError("No binding pocket detected in this structure")

    pocket_res_ids = extract_pocket_residues(atoms, res_ids, pocket)
    volume = len(pocket)
    depth = float(np.max(pocket[:, 2]) - np.min(pocket[:, 2]))
    hydro = hydrophobic_score(residues, pocket_res_ids, res_ids)
    compact = compactness(pocket)
    score = drug_score(volume, depth, hydro)
    decision = classify_decision(score)

    with open(pdb_file) as f:
        pdb_data = f.read()

    return {
        "volume": volume,
        "depth": round(depth, 2),
        "hydrophobic_ratio": round(hydro, 3),
        "compactness": round(compact, 2),
        "drug_score": score,
        "decision": decision,
        "pocket_residues": pocket_res_ids,
        "pdb_data": pdb_data,
        "num_residues": len(res_ids),
        "num_models": len(models),
        "all_clusters": all_clusters,
        "pocket_center": np.mean(pocket, axis=0).tolist(),
    }


# ─────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "StructureAgent API"}


@app.post("/analyze")
async def analyze(
    protein_input: Optional[str] = Form(None),
    pdb_file: Optional[UploadFile] = File(None)
):
    with tempfile.TemporaryDirectory() as work_dir:
        try:
            if pdb_file:
                file_path = os.path.join(work_dir, pdb_file.filename)
                content = await pdb_file.read()
                with open(file_path, "wb") as f:
                    f.write(content)
            elif protein_input:
                file_path = resolve_input(protein_input.strip(), work_dir)
            else:
                raise HTTPException(status_code=400, detail="Provide protein_input or pdb_file")

            result = run_analysis(file_path)
            return JSONResponse(content=result)

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/health")
def health():
    return {"status": "healthy"}
