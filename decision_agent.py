import json
from typing import Dict, Optional, Tuple

from agent_1 import StructureAgent
from bio_agent import (
    get_ensembl_id,
    get_disease_associations,
    find_disease_match,
    parse_evidence_types,
    rule_based_assessment
)
from safety_agent import SafetyAgent


class DecisionAgent:

    def __init__(self, protein_input: str, gene_symbol: str, disease_name: str):
        self.protein_input = protein_input
        self.gene_symbol = gene_symbol
        self.disease_name = disease_name

        self.structure_results = None
        self.biology_results = None
        self.safety_results = None

    # -------------------------------
    # STRUCTURE
    # -------------------------------
    def run_structure(self) -> Optional[Dict]:
        try:
            agent = StructureAgent(self.protein_input)
            result = agent.evaluate()

            if not result:
                return None

            # ✅ FIX: handle visualization correctly
            viz = (
                result.get("visualization")
                or result.get("visualization_html")
                or result.get("visualization_path")
            )

            # ✅ Save visualization to file (important)
            if viz:
                with open("final_visualization.html", "w", encoding="utf-8") as f:
                    f.write(viz)

            result["visualization"] = viz
            result["visualization_file"] = "final_visualization.html" if viz else None

            self.structure_results = result
            return result

        except Exception as e:
            print(f"❌ Structure error: {e}")
            return None

    # -------------------------------
    # BIOLOGY
    # -------------------------------
    def run_biology(self) -> Optional[Dict]:
        try:
            ensembl_id, _ = get_ensembl_id(self.gene_symbol)
            if not ensembl_id:
                return None

            associations = get_disease_associations(ensembl_id)
            if not associations:
                return None

            match = find_disease_match(associations, self.disease_name)
            if not match:
                return None

            evidence = parse_evidence_types(match.get("datatypeScores", []))
            score = match.get("score", 0)

            assessment = rule_based_assessment(score, evidence, self.disease_name)

            result = {
                "overall_score": assessment["overall_score"],
                "verdict": assessment["verdict"],
                "positive_signals": assessment["positive_signals"],
                "warning_signals": assessment["warning_signals"],
                "evidence": evidence,
                "ensembl_id": ensembl_id
            }

            self.biology_results = result
            return result

        except Exception as e:
            print(f"❌ Biology error: {e}")
            return None

    # -------------------------------
    # SAFETY
    # -------------------------------
    def run_safety(self) -> Dict:
        try:
            ensembl_id = self.biology_results.get("ensembl_id")
            agent = SafetyAgent(self.gene_symbol)

            result = agent.run(ensembl_id)

            # ✅ FIX: don’t override blindly
            if "safety_verdict" not in result:
                result["safety_verdict"] = agent._generate_verdict()

            self.safety_results = result
            return result

        except Exception as e:
            print(f"⚠️ Safety fallback: {e}")
            self.safety_results = {
                "safety_index": 0.5,
                "safety_verdict": "UNKNOWN"
            }
            return self.safety_results

    # -------------------------------
    # SCORING (FIXED WEIGHTS)
    # -------------------------------
    def compute_score(self) -> Tuple[float, Dict]:

        s = self.structure_results.get("druggability_score", 0)
        b = self.biology_results.get("overall_score", 0)
        sa = self.safety_results.get("safety_index", 0.5)

        s, b, sa = min(s, 1), min(b, 1), min(sa, 1)

        # ✅ FIX: biology dominant (scientifically correct)
        score = (
            0.30 * s +
            0.50 * b +
            0.20 * sa
        )

        return score, {
            "structure": round(s * 0.30, 3),
            "biology": round(b * 0.50, 3),
            "safety": round(sa * 0.20, 3)
        }

    # -------------------------------
    # VERDICT
    # -------------------------------
    def verdict(self, score: float) -> str:

        # Biology override (critical)
        if "AVOID" in self.biology_results.get("verdict", ""):
            return "🚫 REJECT - weak biological relevance"

        # Safety override
        if "UNSAFE" in self.safety_results.get("safety_verdict", ""):
            return "🚫 REJECT - safety risk"

        if score >= 0.75:
            return "✅ STRONG CANDIDATE"
        elif score >= 0.55:
            return "⭐ MODERATE CANDIDATE"
        elif score >= 0.35:
            return "⚠️ BORDERLINE TARGET"
        else:
            return "🚫 REJECT"

    # -------------------------------
    # RUN PIPELINE
    # -------------------------------
    def run(self) -> Dict:

        print("\n=== DRUG DISCOVERY DECISION PIPELINE ===")

        if not self.run_structure():
            return {"status": "FAILED", "reason": "Structure failed"}

        if not self.run_biology():
            return {"status": "FAILED", "reason": "Biology failed"}

        self.run_safety()

        score, components = self.compute_score()
        final = self.verdict(score)

        return {
            "status": "SUCCESS",
            "protein": self.protein_input,
            "gene": self.gene_symbol,
            "disease": self.disease_name,

            "structure": self.structure_results,
            "biology": self.biology_results,
            "safety": self.safety_results,

            "decision": {
                "integrated_score": round(score, 3),
                "components": components,
                "final_verdict": final
            },

            # ✅ NEW: artifact exposure
            "artifacts": {
                "visualization_file": "final_visualization.html"
            }
        }


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":

    agent = DecisionAgent(
        protein_input="2ITX",
        gene_symbol="EGFR",
        disease_name="breast cancer"
    )

    result = agent.run()

    with open("final_output.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n✓ Saved to final_output.json")
    print("✓ Visualization saved to final_visualization.html")
    print("Gene:", self.gene_symbol)
    print("Disease:", self.disease_name)
    print("Integrated Score:", result["decision"]["integrated_score"]) 
    print("Final Verdict:", result["decision"]["final_verdict"])
