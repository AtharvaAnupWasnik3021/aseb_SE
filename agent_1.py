import numpy as np
import os
import requests
import py3Dmol
from Bio.PDB import PDBParser, PDBList
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN


HYDROPHOBIC = {"ALA","VAL","LEU","ILE","MET","PHE","TRP","PRO"}


class StructureAgent:

    def __init__(self, protein_input):
        self.pdb_file = self.resolve_input(protein_input)
        self.models = []
        self.load_structure()

    # ------------------------------------------------
    # Resolve input (Protein name / PDB ID / File)
    # ------------------------------------------------
    def resolve_input(self, protein_input):

        protein_input = protein_input.strip()

        if os.path.exists(protein_input):
            return protein_input

        clean = protein_input.upper()

        if "PDB" in clean:
            clean = clean.split()[-1]

        if len(clean) == 4 and clean.isalnum():
            return self.download_pdb(clean)

        # Text search fallback
        print("Searching RCSB for:", protein_input)

        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {"value": protein_input}
            },
            "return_type": "entry",
            "request_options": {"pager": {"start": 0, "rows": 1}}
        }

        response = requests.post(
            "https://search.rcsb.org/rcsbsearch/v1/query",
            json=query
        )

        data = response.json()

        if "result_set" not in data or len(data["result_set"]) == 0:
            raise ValueError("No structure found. Try PDB ID directly.")

        pdb_id = data["result_set"][0]["identifier"]
        print("Found PDB:", pdb_id)

        return self.download_pdb(pdb_id)

    def download_pdb(self, pdb_id):

        pdbl = PDBList()
        file_path = pdbl.retrieve_pdb_file(
            pdb_id,
            pdir=".",
            file_format="pdb"
        )

        new_name = f"{pdb_id}.pdb"

        if os.path.exists(new_name):
            return new_name

        os.rename(file_path, new_name)

        return new_name

    # ------------------------------------------------
    # Load Structure
    # ------------------------------------------------
    def load_structure(self):

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", self.pdb_file)

        for model in structure:

            atoms = []
            residues = []
            res_ids = []

            for chain in model:
                prev_res = None

                for residue in chain:

                    if residue.get_resname() == "HOH":
                        continue

                    res_id = residue.get_id()[1]

                    if prev_res and res_id != prev_res + 1:
                        print(f"GAP detected between {prev_res} and {res_id}")

                    prev_res = res_id

                    for atom in residue:
                        if atom.get_parent().id[0] != " ":
                            continue
                        atoms.append(atom.coord)
                        residues.append(residue.get_resname())
                        res_ids.append(res_id)

            self.models.append({
                "atoms": np.array(atoms),
                "residues": residues,
                "res_ids": res_ids
            })

    # ------------------------------------------------
    # Pocket Detection (Improved)
    # ------------------------------------------------
    def detect_pocket(self, atoms):

        tree = KDTree(atoms)
        spacing = 1.0

        min_c = np.min(atoms, axis=0) - 2
        max_c = np.max(atoms, axis=0) + 2

        x = np.arange(min_c[0], max_c[0], spacing)
        y = np.arange(min_c[1], max_c[1], spacing)
        z = np.arange(min_c[2], max_c[2], spacing)

        grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

        cavity = []

        for point in grid:
            dist, _ = tree.query(point)

            if dist > 2.8:
                neighbors = tree.query_ball_point(point, r=5.0)
                if 8 < len(neighbors) < 80:
                    cavity.append(point)

        cavity = np.array(cavity)

        if len(cavity) == 0:
            return None

        clustering = DBSCAN(eps=2.5, min_samples=8).fit(cavity)
        labels = clustering.labels_

        largest = None
        max_size = 0

        for label in set(labels):
            if label == -1:
                continue
            cluster = cavity[labels == label]
            if len(cluster) > max_size:
                largest = cluster
                max_size = len(cluster)

        return largest

    # ------------------------------------------------
    # Extract Pocket Residues
    # ------------------------------------------------
    def extract_pocket_residues(self, atoms, residues, res_ids, pocket):

        tree = KDTree(atoms)
        pocket_res = set()

        for point in pocket:
            neighbors = tree.query_ball_point(point, r=4.0)
            for idx in neighbors:
                pocket_res.add(res_ids[idx])

        return sorted(pocket_res)

    # ------------------------------------------------
    # Evaluate + Visualize
    # ------------------------------------------------
    def evaluate(self):

        print("Total Models:", len(self.models))

        model = self.models[0]

        atoms = model["atoms"]
        residues = model["residues"]
        res_ids = model["res_ids"]

        pocket = self.detect_pocket(atoms)

        if pocket is None:
            print("No druggable pocket detected.")
            return

        pocket_res_ids = self.extract_pocket_residues(
            atoms, residues, res_ids, pocket
        )

        volume = len(pocket)
        depth = np.max(pocket[:,2]) - np.min(pocket[:,2])

        print("Pocket Volume:", volume)
        print("Pocket Depth:", round(depth,2))
        print("Pocket Residue IDs:", pocket_res_ids)

        self.visualize(pocket_res_ids)

    # ------------------------------------------------
    # AlphaFold-style Visualization
    # ------------------------------------------------
    def visualize(self, pocket_res_ids):

        with open(self.pdb_file, "r") as f:
            pdb_data = f.read()

        view = py3Dmol.view(width=900, height=700)
        view.addModel(pdb_data, "pdb")

        # Cartoon structure
        view.setStyle({"cartoon": {"color": "spectrum"}})

        # Highlight pocket residues
        if pocket_res_ids:
            view.setStyle(
                {"resi": pocket_res_ids},
                {"stick": {"colorscheme": "redCarbon"}}
            )

        view.addSurface(py3Dmol.VDW, {"opacity": 0.2})
        view.zoomTo()
        view.spin(True)

        display(view)


# ------------------------------------------------
# MAIN
# ------------------------------------------------
if __name__ == "__main__":

    protein_input = input("Enter Protein Name, PDB ID, or File: ")

    agent = StructureAgent(protein_input)
    viewer = agent.evaluate()
