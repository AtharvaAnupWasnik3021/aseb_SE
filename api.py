import os
import tempfile
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional

from Bio.PDB import PDBParser, PDBList
from scipy.spatial import KDTree, Delaunay
from sklearn.cluster import DBSCAN

app = FastAPI(title="StructureAgent API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}


# -------------------------------
# FILE HANDLING
# -------------------------------
def download_pdb(pdb_id: str, work_dir: str) -> str:
    pdbl = PDBList()
    file_path = pdbl.retrieve_pdb_file(pdb_id, pdir=work_dir, file_format="pdb")
    new_name = os.path.join(work_dir, f"{pdb_id}.pdb")

    if not os.path.exists(new_name) and os.path.exists(file_path):
        os.rename(file_path, new_name)

    return new_name if os.path.exists(new_name) else file_path


def resolve_input(protein_input: str, work_dir: str) -> str:
    clean = protein_input.strip().upper()

    if len(clean) == 4 and clean.isalnum():
        return download_pdb(clean, work_dir)

    raise ValueError("Invalid protein input. Use PDB ID for now.")


# -------------------------------
# STRUCTURE LOADING
# -------------------------------
def load_structure(pdb_file: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    atoms, residues, res_ids = [], [], []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == "HOH":
                    continue
                res_id = residue.get_id()[1]
                for atom in residue:
                    if atom.get_id() == "CA":
                        atoms.append(atom.coord)
                        residues.append(residue.get_resname())
                        res_ids.append(res_id)

    return np.array(atoms), residues, res_ids


# -------------------------------
# POCKET DETECTION
# -------------------------------
def detect_pockets(atoms):
    if len(atoms) < 10:
        return []

    tri = Delaunay(atoms)
    cavities = []

    for simplex in tri.simplices:
        pts = atoms[simplex]
        center = np.mean(pts, axis=0)
        dist = np.min(np.linalg.norm(atoms - center, axis=1))

        if dist > 4:
            cavities.append(center)

    if not cavities:
        return []

    cavities = np.array(cavities)

    clustering = DBSCAN(eps=3, min_samples=6).fit(cavities)
    labels = clustering.labels_

    pockets = []
    for label in set(labels):
        if label == -1:
            continue
        cluster = cavities[labels == label]
        pockets.append(cluster)

    return pockets


# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def compute_features(atoms, residues, res_ids, pocket):
    tree = KDTree(atoms)

    pocket_res = set()
    for point in pocket:
        neighbors = tree.query_ball_point(point, r=6)
        for idx in neighbors:
            pocket_res.add(res_ids[idx])

    pocket_res_ids = sorted(pocket_res)

    # Volume
    volume = len(pocket)

    # 3D depth (correct)
    center = np.mean(pocket, axis=0)
    depth = float(np.max(np.linalg.norm(pocket - center, axis=1)))

    # Hydrophobicity
    pocket_residues = [
        residues[i] for i, r in enumerate(res_ids) if r in pocket_res_ids
    ]
    hydro = (
        sum(1 for r in pocket_residues if r in HYDROPHOBIC)
        / len(pocket_residues)
        if pocket_residues else 0
    )

    # Compactness
    compact = float(np.std(np.linalg.norm(pocket - center, axis=1)))

    return volume, depth, hydro, compact, pocket_res_ids


# -------------------------------
# SCORING
# -------------------------------
def compute_score(volume, depth, hydro, compact):

    # Normalize features
    volume_score = min(volume / 1000, 1.0)
    depth_score = min(depth / 30, 1.0)
    compact_score = max(0, 1 - compact / 10)

    score = (
        0.3 * volume_score +
        0.3 * depth_score +
        0.25 * hydro +
        0.15 * compact_score
    )

    # Penalize overly large pockets
    if volume > 2000:
        score *= 0.7

    return round(min(score, 1.0), 3)


def classify(score):
    if score >= 0.7:
        return "Likely Druggable"
    elif score >= 0.5:
        return "Moderate Potential"
    else:
        return "Poor Drug Target"


# -------------------------------
# MAIN ANALYSIS
# -------------------------------
def run_analysis(pdb_file: str):

    atoms, residues, res_ids = load_structure(pdb_file)

    pockets = detect_pockets(atoms)

    if not pockets:
        return {
            "status": "NO_POCKET",
            "drug_score": 0,
            "decision": "No binding pocket detected"
        }

    best_result = None
    best_score = -1

    for pocket in pockets:
        volume, depth, hydro, compact, res_ids_pocket = compute_features(
            atoms, residues, res_ids, pocket
        )

        score = compute_score(volume, depth, hydro, compact)

        if score > best_score:
            best_score = score
            best_result = {
                "volume": volume,
                "depth": round(depth, 2),
                "hydrophobic_ratio": round(hydro, 3),
                "compactness": round(compact, 2),
                "drug_score": score,
                "decision": classify(score),
                "pocket_residues": res_ids_pocket,
                "pocket_center": np.mean(pocket, axis=0).tolist(),
            }

    return best_result


# -------------------------------
# API
# -------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "StructureAgent v2"}


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
                file_path = resolve_input(protein_input, work_dir)

            else:
                raise HTTPException(status_code=400, detail="Provide input")

            result = run_analysis(file_path)
            return JSONResponse(content=result)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy"}