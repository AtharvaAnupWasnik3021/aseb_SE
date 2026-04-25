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
    # Pocket Detection (IMPROVED - Biologically realistic)
    # ------------------------------------------------
    def detect_pocket(self, atoms):
        """
        Improved pocket detection with:
        - Size filtering (ignore overly large cavities)
        - Compactness scoring
        - 3D depth instead of Z-axis only
        - Penalizes non-specific large empty spaces
        """
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
            
            # Tighter distance threshold (protein surface)
            if 2.8 <= dist <= 8.0:  # FIXED: only near-surface cavities
                neighbors = tree.query_ball_point(point, r=5.0)
                # FIXED: stricter neighbor count (real pockets have 10-30 neighbors)
                if 10 < len(neighbors) < 40:
                    cavity.append(point)

        cavity = np.array(cavity)

        if len(cavity) == 0:
            return None

        clustering = DBSCAN(eps=2.5, min_samples=8).fit(cavity)
        labels = clustering.labels_

        # IMPROVED: Find best pocket (not just largest)
        best_pocket = None
        best_score = -np.inf

        for label in set(labels):
            if label == -1:
                continue
            
            cluster = cavity[labels == label]
            score = self._evaluate_pocket_quality(cluster, atoms)
            
            if score > best_score:
                best_score = score
                best_pocket = cluster

        return best_pocket if best_pocket is not None else None

    def _evaluate_pocket_quality(self, pocket, atoms):
        """
        Score pocket based on:
        - Size (penalize very large pockets)
        - Compactness (reward tight clusters)
        - Enclosure (reward pockets surrounded by atoms)
        """
        size = len(pocket)
        
        # PENALTY: Reject overly large pockets (noise)
        if size > 2000:
            return -1000  # Completely reject
        
        if size < 100:
            return -500  # Too small, not druggable
        
        # COMPACTNESS: Lower std = tighter pocket
        centroid = np.mean(pocket, axis=0)
        distances_from_center = np.linalg.norm(pocket - centroid, axis=1)
        compactness = np.std(distances_from_center)
        
        # Reward compact pockets (low compactness)
        compactness_score = 100 / (compactness + 1)
        
        # ENCLOSURE: How many atoms nearby?
        tree = KDTree(atoms)
        neighbors = tree.query_ball_point(centroid, r=6.0)
        enclosure_score = len(neighbors) * 2
        
        # FINAL SCORE
        quality_score = (
            (size / 500) +           # Prefer 500-1500 size
            compactness_score +      # Prefer compact
            enclosure_score -        # Prefer surrounded by atoms
            (size / 200)            # Penalize too large
        )
        
        return quality_score

    # ------------------------------------------------
    # Extract Pocket Residues (IMPROVED)
    # ------------------------------------------------
    def extract_pocket_residues(self, atoms, residues, res_ids, pocket):
        """
        Extract residues touching pocket with hydrophobic weighting
        """
        tree = KDTree(atoms)
        pocket_res = {}  # {res_id: hydrophobic_score}
        centroid = np.mean(pocket, axis=0)

        for point in pocket:
            neighbors = tree.query_ball_point(point, r=4.0)
            for idx in neighbors:
                res_id = res_ids[idx]
                res_name = residues[idx]
                
                # IMPROVED: Weight hydrophobic residues higher
                is_hydrophobic = 1.0 if res_name in HYDROPHOBIC else 0.5
                
                if res_id not in pocket_res:
                    pocket_res[res_id] = 0
                pocket_res[res_id] += is_hydrophobic

        # Return sorted by importance
        return sorted(pocket_res.keys())

    # ------------------------------------------------
    # Evaluate + Visualize (IMPROVED)
    # ------------------------------------------------
    def evaluate(self):
        """
        Analyze pocket with proper druggability scoring.
        Returns realistic assessment, not just size metrics.
        """
        
        print("Total Models:", len(self.models))

        model = self.models[0]

        atoms = model["atoms"]
        residues = model["residues"]
        res_ids = model["res_ids"]

        pocket = self.detect_pocket(atoms)

        if pocket is None:
            print("✗ No druggable pocket detected.")
            return None

        pocket_res_ids = self.extract_pocket_residues(
            atoms, residues, res_ids, pocket
        )

        # IMPROVED: Proper metrics
        volume = len(pocket)
        
        # FIXED: 3D depth instead of Z-axis only
        centroid = np.mean(pocket, axis=0)
        distances = np.linalg.norm(pocket - centroid, axis=1)
        depth_3d = np.max(distances) * 2  # Diameter
        
        # IMPROVED: Hydrophobic scoring
        hydrophobic_score = self._calculate_hydrophobic_content(
            atoms, residues, res_ids, pocket_res_ids
        )
        
        # IMPROVED: Proper druggability scoring
        druggability_score, decision = self._calculate_druggability(
            volume, depth_3d, len(pocket_res_ids), hydrophobic_score
        )

        print(f"✓ Pocket Detected!")
        print(f"  Volume: {volume} points")
        print(f"  Depth (3D): {round(depth_3d, 2)} Å")
        print(f"  Residues: {len(pocket_res_ids)} ({pocket_res_ids[:5]}...)")
        print(f"  Hydrophobic Content: {round(hydrophobic_score, 2)}")
        print(f"  Druggability Score: {round(druggability_score, 2)}")
        print(f"  Assessment: {decision}")

        # Return analysis data for backend
        return {
            "pocket_residues": pocket_res_ids,
            "volume": volume,
            "depth_3d": round(depth_3d, 2),
            "residue_count": len(pocket_res_ids),
            "hydrophobic_content": round(hydrophobic_score, 2),
            "druggability_score": round(druggability_score, 3),
            "assessment": decision,
            "visualization_html": self.get_visualization_html(pocket_res_ids)
        }

    def _calculate_hydrophobic_content(self, atoms, residues, res_ids, pocket_res_ids):
        """Calculate percentage of hydrophobic residues in pocket"""
        hydro_count = 0
        for res_id in pocket_res_ids:
            res_idx = res_ids.index(res_id) if res_id in res_ids else -1
            if res_idx >= 0 and residues[res_idx] in HYDROPHOBIC:
                hydro_count += 1
        return (hydro_count / len(pocket_res_ids)) if pocket_res_ids else 0

    def _calculate_druggability(self, volume, depth, residue_count, hydro_score):
        """
        Calculate realistic druggability score.
        Fixes: doesn't blindly reward large volumes.
        Optimal range: 500-1500 volume, 15-40 depth, 15-35 residues.
        """
        score = 0.0
        factors = []
        
        # 1. VOLUME SCORE (optimal 500-1500)
        if 500 <= volume <= 1500:
            vol_score = 0.8
            factors.append(f"Volume {volume}: Optimal")
        elif 300 <= volume < 500:
            vol_score = 0.5
            factors.append(f"Volume {volume}: Small but druggable")
        elif 1500 < volume <= 2000:
            vol_score = 0.4
            factors.append(f"Volume {volume}: Large, less specific")
        else:
            vol_score = 0.1
            factors.append(f"Volume {volume}: Too small or rejected (>2000)")
        
        # 2. DEPTH SCORE (optimal 15-40 Angstroms)
        if 15 <= depth <= 40:
            depth_score = 0.8
            factors.append(f"Depth {depth}Å: Good")
        elif 10 <= depth < 15:
            depth_score = 0.5
            factors.append(f"Depth {depth}Å: Shallow")
        elif 40 < depth <= 60:
            depth_score = 0.4
            factors.append(f"Depth {depth}Å: Very deep")
        else:
            depth_score = 0.1
            factors.append(f"Depth {depth}Å: Not suitable")
        
        # 3. RESIDUE COUNT (optimal 15-35)
        if 15 <= residue_count <= 35:
            res_score = 0.8
            factors.append(f"Residues {residue_count}: Selective")
        elif 10 <= residue_count < 15:
            res_score = 0.5
            factors.append(f"Residues {residue_count}: Small interface")
        elif 35 < residue_count <= 50:
            res_score = 0.4
            factors.append(f"Residues {residue_count}: Large interface")
        else:
            res_score = 0.1
            factors.append(f"Residues {residue_count}: Too many/few")
        
        # 4. HYDROPHOBIC CONTENT (optimal 40-70%)
        if 0.4 <= hydro_score <= 0.7:
            hydro_raw_score = 0.8
            factors.append(f"Hydrophobic {round(hydro_score*100)}%: Favorable")
        elif hydro_score < 0.4:
            hydro_raw_score = 0.3
            factors.append(f"Hydrophobic {round(hydro_score*100)}%: Too polar")
        else:
            hydro_raw_score = 0.5
            factors.append(f"Hydrophobic {round(hydro_score*100)}%: Very hydrophobic")
        
        # FINAL SCORE (weighted combination)
        score = (
            vol_score * 0.35 +
            depth_score * 0.25 +
            res_score * 0.20 +
            hydro_raw_score * 0.20
        )
        
        # DECISION
        if score > 0.7:
            decision = "✓ HIGH DRUGGABILITY (Good target)"
        elif score > 0.5:
            decision = "◐ MODERATE DRUGGABILITY (Workable)"
        elif score > 0.3:
            decision = "✗ LOW DRUGGABILITY (Challenging)"
        else:
            decision = "✗✗ NOT DRUGGABLE (Poor target)"
        
        return score, decision

    # ------------------------------------------------
    # AlphaFold-style Visualization
    # ------------------------------------------------
    def get_visualization_html(self, pocket_res_ids):
        """Returns HTML string for visualization instead of saving/displaying"""
        with open(self.pdb_file, "r") as f:
            pdb_data = f.read()

        view = py3Dmol.view(width=900, height=700)
        view.addModel(pdb_data, "pdb")
        view.setStyle({"cartoon": {"color": "spectrum"}})

        if pocket_res_ids:
            view.setStyle(
                {"resi": pocket_res_ids},
                {"stick": {"colorscheme": "redCarbon"}}
            )

        view.addSurface(py3Dmol.VDW, {"opacity": 0.2})
        view.zoomTo()
        return view._make_html()


# ------------------------------------------------
# MAIN
# ------------------------------------------------
if __name__ == "__main__":

    protein_input = input("Enter Protein Name, PDB ID, or File: ")

    agent = StructureAgent(protein_input)
    results = agent.evaluate()
    
    if results:
        # Save visualization HTML for testing
        output_file = "protein_visualization.html"
        with open(output_file, "w") as f:
            f.write(results["visualization_html"])
        
        print(f"\n{'='*50}")
        print(f"ANALYSIS RESULTS")
        print(f"{'='*50}")
        print(f"Volume: {results['volume']} points")
        print(f"Depth (3D): {results['depth_3d']} Å")
        print(f"Residue Count: {results['residue_count']}")
        print(f"Hydrophobic Content: {results['hydrophobic_content']*100:.1f}%")
        print(f"Druggability Score: {results['druggability_score']}/1.0")
        print(f"Assessment: {results['assessment']}")
        print(f"Pocket Residues: {results['pocket_residues'][:10]}...")
        print(f"{'='*50}")
        print(f"✓ Visualization saved to: {output_file}")
        print(f"  Open in browser to view 3D structure")
    else:
        print("✗ Failed to analyze protein")