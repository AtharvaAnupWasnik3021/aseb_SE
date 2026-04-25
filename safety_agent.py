import requests
from typing import Dict, List


class SafetyAgent:

    OPEN_TARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    def __init__(self, gene_symbol: str):
        self.gene_symbol = gene_symbol
        self.ensembl_id = None
        self.results = {}

    # -------------------------------
    # API CALLS
    # -------------------------------
    def _query(self, query, variables):
        try:
            res = requests.post(
                self.OPEN_TARGETS_URL,
                json={"query": query, "variables": variables},
                timeout=10
            )
            return res.json()
        except Exception:
            return {}

    # -------------------------------
    # TISSUE EXPRESSION
    # -------------------------------
    def get_tissue_expression(self):
        query = """
        query GetExpression($id: String!) {
          target(ensemblId: $id) {
            proteinExpression {
              evidence {
                tissue { name }
                level
              }
            }
          }
        }
        """
        data = self._query(query, {"id": self.ensembl_id})
        return data.get("data", {}).get("target", {}).get("proteinExpression", {})

    def analyze_tissue_expression(self, data):
        critical = ["heart", "brain", "liver", "kidney"]
        risks = []
        penalty = 0

        evidence = data.get("evidence", [])

        for item in evidence:
            tissue = item.get("tissue", {}).get("name", "").lower()
            level = item.get("level", "").lower()

            if any(c in tissue for c in critical) and level in ["high", "medium"]:
                risks.append({
                    "tissue": tissue,
                    "level": level
                })
                penalty += 0.1 if level == "medium" else 0.2

        score = max(0.0, 1.0 - penalty)

        return {
            "score": score,
            "risks": risks,
            "has_data": len(evidence) > 0
        }

    # -------------------------------
    # OFF TARGET
    # -------------------------------
    def assess_off_target(self):
        query = """
        query GetSimilar($id: String!) {
          target(ensemblId: $id) {
            similarEntities(size: 10) {
              results { score }
            }
          }
        }
        """
        data = self._query(query, {"id": self.ensembl_id})
        results = data.get("data", {}).get("target", {}) \
                      .get("similarEntities", {}).get("results", [])

        if not results:
            return {"score": 0.5, "has_data": False}

        avg_similarity = sum(x.get("score", 0) for x in results) / len(results)

        return {
            "score": 1.0 - avg_similarity,
            "has_data": True
        }

    # -------------------------------
    # ADVERSE EVENTS
    # -------------------------------
    def get_adverse_events(self):
        query = """
        query GetAdverse($id: String!) {
          target(ensemblId: $id) {
            adverseEvents {
              events { name count }
            }
          }
        }
        """
        data = self._query(query, {"id": self.ensembl_id})
        return data.get("data", {}).get("target", {}) \
                   .get("adverseEvents", {}).get("events", [])

    def analyze_adverse(self, events):
        if not events:
            return {"score": 0.5, "has_data": False}

        severity = 0

        for e in events:
            name = e.get("name", "").lower()

            if any(x in name for x in ["fatal", "death"]):
                severity = max(severity, 1.0)
            elif any(x in name for x in ["cardiac", "hepatic"]):
                severity = max(severity, 0.8)
            elif any(x in name for x in ["bleeding", "renal"]):
                severity = max(severity, 0.6)
            else:
                severity = max(severity, 0.4)

        return {
            "score": 1.0 - severity,
            "has_data": True
        }

    # -------------------------------
    # SAFETY INDEX
    # -------------------------------
    def compute_safety_index(self, tissue, off_target, adverse):

        # detect if we actually have data
        data_flags = [
            tissue["has_data"],
            off_target["has_data"],
            adverse["has_data"]
        ]

        if not any(data_flags):
            return 0.5, "⚠️ UNKNOWN - insufficient safety data"

        # weighted score
        score = (
            tissue["score"] * 0.35 +
            off_target["score"] * 0.35 +
            adverse["score"] * 0.30
        )

        if score >= 0.8:
            verdict = "✅ SAFE"
        elif score >= 0.6:
            verdict = "⚠️ MODERATE RISK"
        elif score >= 0.4:
            verdict = "⚠️ HIGH RISK"
        else:
            verdict = "🚫 UNSAFE"

        return score, verdict

    # -------------------------------
    # MAIN
    # -------------------------------
    def run(self, ensembl_id: str):

        self.ensembl_id = ensembl_id

        # Pathogen override (critical)
        if any(x in self.gene_symbol.lower() for x in ["pol", "env", "gag"]):
            return {
                "safety_index": 0.7,
                "safety_interpretation": "⚠️ PATHOGEN TARGET - host toxicity less relevant",
                "tissue_analysis": {"tissue_risks": []},
                "off_target_effects": {"selectivity_risk": "LOW"},
                "side_effects": {"critical_events": []}
            }

        tissue = self.analyze_tissue_expression(self.get_tissue_expression())
        off_target = self.assess_off_target()
        adverse = self.analyze_adverse(self.get_adverse_events())

        score, interpretation = self.compute_safety_index(tissue, off_target, adverse)

        self.results = {
            "safety_index": round(score, 3),
            "safety_interpretation": interpretation,
            "tissue_analysis": {
                "tissue_risks": tissue["risks"]
            },
            "off_target_effects": {
                "selectivity_risk": "LOW" if off_target["score"] > 0.6 else
                                    "MEDIUM" if off_target["score"] > 0.3 else "HIGH"
            },
            "side_effects": {
                "critical_events": []
            }
        }

        return self.results

    def _generate_verdict(self):
        score = self.results.get("safety_index", 0.5)

        if score >= 0.75:
            return "✅ SAFE FOR DEVELOPMENT"
        elif score >= 0.6:
            return "⚠️ PROCEED WITH CAUTION"
        elif score >= 0.4:
            return "🚫 HIGH RISK"
        else:
            return "🚫 UNSAFE"