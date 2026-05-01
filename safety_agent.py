import requests
from typing import Dict, List


class SafetyAgent:

    OPEN_TARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
    CHEMBL_URL = "https://www.ebi.ac.uk/chembl/api/data"

    def __init__(self, gene_symbol: str):
        self.gene_symbol = gene_symbol.upper()
        self.ensembl_id = None
        self.results = {}

    # -------------------------------
    # GENERIC API CALL
    # -------------------------------
    def _safe_get(self, url, params=None):
        try:
            r = requests.get(url, params=params, timeout=10)
            return r.json()
        except Exception:
            return {}

    def _safe_post(self, url, payload):
        try:
            r = requests.post(url, json=payload, timeout=10)
            return r.json()
        except Exception:
            return {}

    # -------------------------------
    # STEP 1: GET ENSEMBL ID
    # -------------------------------
    def get_ensembl_id(self):
        query = """
        query ($symbol: String!) {
          search(queryString: $symbol, entityNames: ["target"]) {
            hits {
              object {
                ... on Target {
                  id
                  approvedSymbol
                }
              }
            }
          }
        }
        """
        data = self._safe_post(self.OPEN_TARGETS_URL, {
            "query": query,
            "variables": {"symbol": self.gene_symbol}
        })

        hits = data.get("data", {}).get("search", {}).get("hits", [])
        for hit in hits:
            obj = hit.get("object", {})
            if obj.get("approvedSymbol", "").upper() == self.gene_symbol:
                return obj["id"]

        return None

    # -------------------------------
    # STEP 2: TISSUE RISK (OpenTargets fallback)
    # -------------------------------
    def tissue_risk(self):
        query = """
        query ($id: String!) {
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

        data = self._safe_post(self.OPEN_TARGETS_URL, {
            "query": query,
            "variables": {"id": self.ensembl_id}
        })

        evidence = data.get("data", {}).get("target", {}) \
                       .get("proteinExpression", {}) \
                       .get("evidence", [])

        if not evidence:
            return None

        critical = {"brain": 0.4, "heart": 0.4, "liver": 0.3, "kidney": 0.3}
        level_weight = {"high": 1.0, "medium": 0.6, "low": 0.2}

        penalty = 0

        for item in evidence:
            tissue = item.get("tissue", {}).get("name", "").lower()
            level = item.get("level", "").lower()

            for key in critical:
                if key in tissue:
                    penalty += critical[key] * level_weight.get(level, 0.2)

        return max(0, 1 - penalty)

    # -------------------------------
    # STEP 3: GET CHEMBL TARGET
    # -------------------------------
    def get_chembl_target(self):

        url = f"{self.CHEMBL_URL}/target/search.json"
        data = self._safe_get(url, {"q": self.gene_symbol})

        targets = data.get("targets", [])
        if not targets:
            return None

        return targets[0]["target_chembl_id"]

    # -------------------------------
    # STEP 4: GET DRUGS
    # -------------------------------
    def get_drugs(self, chembl_id):

        url = f"{self.CHEMBL_URL}/activity.json"
        data = self._safe_get(url, {"target_chembl_id": chembl_id, "limit": 20})

        return data.get("activities", [])

    # -------------------------------
    # STEP 5: ADVERSE RISK (FROM DRUGS)
    # -------------------------------
    def adverse_risk(self, activities):

        if not activities:
            return None

        # crude proxy: strong binding drugs → higher side effect risk
        affinities = []

        for act in activities:
            val = act.get("standard_value")
            if val:
                try:
                    affinities.append(float(val))
                except:
                    continue

        if not affinities:
            return None

        avg_affinity = sum(affinities) / len(affinities)

        # normalize (lower IC50 = stronger → higher risk)
        if avg_affinity < 100:
            risk = 0.8
        elif avg_affinity < 1000:
            risk = 0.6
        else:
            risk = 0.4

        return 1 - risk

    # -------------------------------
    # STEP 6: OFF-TARGET (SIMILARITY)
    # -------------------------------
    def off_target_risk(self):

        query = """
        query ($id: String!) {
          target(ensemblId: $id) {
            similarEntities(size: 20) {
              results { score }
            }
          }
        }
        """

        data = self._safe_post(self.OPEN_TARGETS_URL, {
            "query": query,
            "variables": {"id": self.ensembl_id}
        })

        results = data.get("data", {}).get("target", {}) \
                      .get("similarEntities", {}).get("results", [])

        if not results:
            return None

        scores = [x.get("score", 0) for x in results]

        avg = sum(scores) / len(scores)
        return max(0, 1 - avg)

    # -------------------------------
    # FINAL SAFETY INDEX
    # -------------------------------
    def compute(self, tissue, off_target, adverse):

        components = []

        if tissue is not None:
            components.append((tissue, 0.4))

        if off_target is not None:
            components.append((off_target, 0.3))

        if adverse is not None:
            components.append((adverse, 0.3))

        if not components:
            return 0.5, "UNKNOWN"

        score = sum(s * w for s, w in components) / sum(w for _, w in components)

        if score > 0.75:
            verdict = "SAFE"
        elif score > 0.55:
            verdict = "MODERATE RISK"
        elif score > 0.35:
            verdict = "HIGH RISK"
        else:
            verdict = "UNSAFE"

        return score, verdict

    # -------------------------------
    # MAIN RUN
    # -------------------------------
    def run(self):

        self.ensembl_id = self.get_ensembl_id()
        if not self.ensembl_id:
            return {"error": "Gene not found"}

        tissue = self.tissue_risk()

        chembl_id = self.get_chembl_target()
        activities = self.get_drugs(chembl_id) if chembl_id else []

        adverse = self.adverse_risk(activities)
        off_target = self.off_target_risk()

        score, verdict = self.compute(tissue, off_target, adverse)

        return {
            "safety_index": round(score, 3),
            "safety_verdict": verdict,
            "components": {
                "tissue": tissue,
                "off_target": off_target,
                "adverse": adverse
            }
        }