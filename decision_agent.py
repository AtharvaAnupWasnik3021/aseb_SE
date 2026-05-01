import json
from typing import Dict, Optional, Tuple

from structure_agent import StructureAgent
from biology_agent import BiologyAgent
from chemistry_agent import ChemistryAgent
from safety_agent import SafetyAgent


class DecisionAgent:
    """Runs four independent agents and makes the final target decision."""

    def __init__(self, protein_input: str, gene_symbol: str, disease_name: str):
        self.protein_input = protein_input
        self.gene_symbol = gene_symbol
        self.disease_name = disease_name

        self.structure_results = None
        self.biology_results = None
        self.safety_results = None
        self.chemistry_results = None

    def run_structure(self) -> Optional[Dict]:
        try:
            agent = StructureAgent(self.protein_input)
            result = agent.evaluate()
            if not result:
                return None

            viz = (
                result.get("visualization")
                or result.get("visualization_html")
                or result.get("visualization_path")
            )
            if viz:
                with open("final_visualization.html", "w", encoding="utf-8") as f:
                    f.write(viz)

            result.pop("visualization_html", None)
            result["visualization"] = None
            result["visualization_file"] = "final_visualization.html" if viz else None
            self.structure_results = result
            return result
        except Exception as exc:
            print(f"Structure agent failed: {exc}")
            return None

    def run_biology(self) -> Optional[Dict]:
        try:
            result = BiologyAgent(self.gene_symbol, self.disease_name).run()
            if result.get("error"):
                print(f"Biology agent failed: {result['error']}")
                return None

            self.biology_results = result
            return result
        except Exception as exc:
            print(f"Biology agent failed: {exc}")
            return None

    def run_safety(self) -> Dict:
        try:
            ensembl_id = (self.biology_results or {}).get("ensembl_id")
            result = SafetyAgent(self.gene_symbol).run(ensembl_id=ensembl_id)
            if result.get("error"):
                result = {
                    "safety_index": 0.5,
                    "safety_verdict": "UNKNOWN",
                    "components": {},
                    "warning_signals": [result["error"]],
                }

            self.safety_results = result
            return result
        except Exception as exc:
            print(f"Safety agent fallback: {exc}")
            self.safety_results = {
                "safety_index": 0.5,
                "safety_verdict": "UNKNOWN",
                "components": {},
                "warning_signals": [str(exc)],
            }
            return self.safety_results

    def run_chemistry(self) -> Dict:
        try:
            result = ChemistryAgent(
                pdb_id=self.protein_input,
                gene_symbol=self.gene_symbol
            ).run()
            self.chemistry_results = result
            return result
        except Exception as exc:
            print(f"Chemistry agent fallback: {exc}")
            self.chemistry_results = {
                "chemistry_score": 0.5,
                "chemistry_verdict": "UNKNOWN",
                "warning_signals": [str(exc)],
                "sample_results": [],
            }
            return self.chemistry_results

    def compute_score(self) -> Tuple[float, Dict]:
        structure = min((self.structure_results or {}).get("druggability_score", 0), 1)
        biology = min((self.biology_results or {}).get("overall_score", 0), 1)
        safety = min((self.safety_results or {}).get("safety_index", 0.5), 1)
        chemistry = min((self.chemistry_results or {}).get("chemistry_score", 0.5), 1)

        score = (
            0.25 * structure +
            0.40 * biology +
            0.20 * safety +
            0.15 * chemistry
        )

        return score, {
            "structure": round(structure * 0.25, 3),
            "biology": round(biology * 0.40, 3),
            "safety": round(safety * 0.20, 3),
            "chemistry": round(chemistry * 0.15, 3),
        }

    def final_verdict(self, score: float) -> str:
        biology_verdict = (self.biology_results or {}).get("verdict", "")
        safety_verdict = (self.safety_results or {}).get("safety_verdict", "")
        chemistry_verdict = (self.chemistry_results or {}).get("chemistry_verdict", "")

        if "AVOID" in biology_verdict:
            return "REJECT - weak biological relevance"
        if "UNSAFE" in safety_verdict:
            return "REJECT - safety risk"
        if "POOR" in chemistry_verdict:
            return "REJECT - poor chemistry feasibility"

        if score >= 0.75:
            return "USE - strong drug target candidate"
        if score >= 0.55:
            return "CONSIDER - moderate drug target candidate"
        if score >= 0.35:
            return "HOLD - borderline target, needs more validation"
        return "REJECT - low integrated confidence"

    def recommendation(self, final_verdict: str) -> str:
        if final_verdict.startswith("USE"):
            return "The chosen protein should be used for drug discovery follow-up."
        if final_verdict.startswith("CONSIDER"):
            return (
                "The chosen protein can be considered for drug discovery, but it "
                "should be validated further before committing major resources."
            )
        if final_verdict.startswith("HOLD"):
            return (
                "The chosen protein should not be advanced yet; collect stronger "
                "biology, safety, or chemistry evidence first."
            )
        return "The chosen protein should not be used for drug creation at this stage."

    def summary(self) -> Dict:
        return {
            "structure": {
                "score": (self.structure_results or {}).get("druggability_score"),
                "assessment": (self.structure_results or {}).get("assessment"),
                "key_detail": f"{(self.structure_results or {}).get('residue_count')} pocket residues",
            },
            "biology": {
                "score": (self.biology_results or {}).get("overall_score"),
                "verdict": (self.biology_results or {}).get("verdict"),
                "disease": (self.biology_results or {}).get("disease"),
            },
            "safety": {
                "score": (self.safety_results or {}).get("safety_index"),
                "verdict": (self.safety_results or {}).get("safety_verdict"),
            },
            "chemistry": {
                "score": (self.chemistry_results or {}).get("chemistry_score"),
                "verdict": (self.chemistry_results or {}).get("chemistry_verdict"),
                "activity_count": (self.chemistry_results or {}).get("activity_count"),
            },
        }

    def run(self) -> Dict:
        print("\n=== DRUG DISCOVERY DECISION PIPELINE ===")

        if not self.run_structure():
            return {"status": "FAILED", "reason": "Structure agent failed"}

        if not self.run_biology():
            return {"status": "FAILED", "reason": "Biology agent failed"}

        self.run_safety()
        self.run_chemistry()

        score, components = self.compute_score()
        verdict = self.final_verdict(score)

        return {
            "status": "SUCCESS",
            "protein": self.protein_input,
            "gene": self.gene_symbol,
            "disease": self.disease_name,
            "agents": {
                "structure": self.structure_results,
                "biology": self.biology_results,
                "safety": self.safety_results,
                "chemistry": self.chemistry_results,
            },
            "summary": self.summary(),
            "decision": {
                "integrated_score": round(score, 3),
                "components": components,
                "final_verdict": verdict,
                "recommendation": self.recommendation(verdict),
            },
            "artifacts": {
                "visualization_file": "final_visualization.html"
            }
        }


if __name__ == "__main__":
    agent = DecisionAgent(
        protein_input="2ITX",
        gene_symbol="EGFR",
        disease_name="breast cancer"
    )

    result = agent.run()

    with open("final_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nSaved to final_output.json")
    print("Visualization saved to final_visualization.html")
    print("Gene:", result.get("gene"))
    print("Disease:", result.get("disease"))
    if result.get("status") == "SUCCESS":
        print("Integrated Score:", result["decision"]["integrated_score"])
        print("Final Verdict:", result["decision"]["final_verdict"])
        print("Recommendation:", result["decision"]["recommendation"])
    else:
        print("Status:", result["status"])
        print("Reason:", result.get("reason"))
