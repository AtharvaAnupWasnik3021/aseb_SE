import math
from typing import Dict, List, Optional

import requests


class ChemistryAgent:
    """Independent chemistry feasibility agent using RCSB and ChEMBL metadata."""

    RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    RCSB_POLYMER_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
    CHEMBL_URL = "https://www.ebi.ac.uk/chembl/api/data"

    def __init__(self, pdb_id: str, gene_symbol: Optional[str] = None, limit: int = 100):
        self.pdb_id = pdb_id.strip().upper()
        self.gene_symbol = gene_symbol.upper() if gene_symbol else None
        self.limit = limit

    def _safe_get(self, url, params=None):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception:
            return {}

    def _pdb_to_uniprot(self) -> Optional[str]:
        entry = self._safe_get(self.RCSB_ENTRY_URL.format(pdb_id=self.pdb_id))
        entity_ids = entry.get("rcsb_entry_container_identifiers", {}).get(
            "polymer_entity_ids", []
        )

        for entity_id in entity_ids:
            entity = self._safe_get(
                self.RCSB_POLYMER_URL.format(
                    pdb_id=self.pdb_id,
                    entity_id=entity_id
                )
            )
            refs = entity.get("rcsb_polymer_entity_container_identifiers", {})
            accessions = refs.get("uniprot_ids", [])
            if accessions:
                return accessions[0]

        return None

    def _uniprot_to_chembl(self, uniprot_id: str) -> Optional[str]:
        data = self._safe_get(
            f"{self.CHEMBL_URL}/target.json",
            {"target_components__accession": uniprot_id}
        )
        targets = data.get("targets", [])
        if targets:
            return targets[0].get("target_chembl_id")
        return None

    def _gene_to_chembl(self) -> Optional[str]:
        if not self.gene_symbol:
            return None

        data = self._safe_get(
            f"{self.CHEMBL_URL}/target/search.json",
            {"q": self.gene_symbol}
        )
        targets = data.get("targets", [])
        if targets:
            return targets[0].get("target_chembl_id")
        return None

    def _fetch_activities(self, chembl_id: str) -> List[Dict]:
        data = self._safe_get(
            f"{self.CHEMBL_URL}/activity.json",
            {
                "target_chembl_id": chembl_id,
                "standard_type__in": "IC50,Ki,Kd,EC50",
                "limit": self.limit,
            }
        )
        return data.get("activities", [])

    def _fetch_molecule(self, molecule_chembl_id: str) -> Dict:
        return self._safe_get(
            f"{self.CHEMBL_URL}/molecule/{molecule_chembl_id}.json"
        )

    @staticmethod
    def _binding_score(value) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if value <= 10:
            return 1.0
        if value <= 100:
            return 0.85
        if value <= 1000:
            return 0.65
        if value <= 10000:
            return 0.35
        return 0.1

    @staticmethod
    def _property_score(properties: Dict) -> float:
        if not properties:
            return 0.5

        mw = ChemistryAgent._as_float(properties.get("full_mwt"))
        logp = ChemistryAgent._as_float(properties.get("alogp"))
        hba = ChemistryAgent._as_float(properties.get("hba"))
        hbd = ChemistryAgent._as_float(properties.get("hbd"))
        psa = ChemistryAgent._as_float(properties.get("psa"))

        checks = []
        if mw is not None:
            checks.append(mw <= 500)
        if logp is not None:
            checks.append(logp <= 5)
        if hba is not None:
            checks.append(hba <= 10)
        if hbd is not None:
            checks.append(hbd <= 5)
        if psa is not None:
            checks.append(psa <= 140)

        if not checks:
            return 0.5

        return sum(1 for check in checks if check) / len(checks)

    @staticmethod
    def _as_float(value):
        try:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _score_activities(self, activities: List[Dict]) -> Dict:
        samples = []

        for activity in activities:
            molecule_id = activity.get("molecule_chembl_id")
            if not molecule_id:
                continue

            molecule = self._fetch_molecule(molecule_id)
            properties = molecule.get("molecule_properties") or {}
            binding = self._binding_score(activity.get("standard_value"))
            drug_like = self._property_score(properties)
            score = 0.65 * binding + 0.35 * drug_like

            samples.append({
                "molecule_chembl_id": molecule_id,
                "standard_type": activity.get("standard_type"),
                "standard_value": activity.get("standard_value"),
                "standard_units": activity.get("standard_units"),
                "binding_score": round(binding, 3),
                "drug_likeness": round(drug_like, 3),
                "score": round(score, 3),
            })

            if len(samples) >= 20:
                break

        if not samples:
            return {
                "chemistry_score": 0.5,
                "chemistry_verdict": "UNKNOWN",
                "sample_results": [],
                "warning_signals": ["No usable ligand activity data found"],
            }

        chemistry_score = sum(item["score"] for item in samples) / len(samples)

        if chemistry_score >= 0.75:
            verdict = "HIGH FEASIBILITY"
        elif chemistry_score >= 0.5:
            verdict = "MODERATE FEASIBILITY"
        elif chemistry_score >= 0.3:
            verdict = "LOW FEASIBILITY"
        else:
            verdict = "POOR FEASIBILITY"

        return {
            "chemistry_score": round(chemistry_score, 3),
            "chemistry_verdict": verdict,
            "sample_results": samples[:5],
            "warning_signals": [],
        }

    def run(self) -> Dict:
        uniprot_id = self._pdb_to_uniprot()
        chembl_id = self._uniprot_to_chembl(uniprot_id) if uniprot_id else None
        chembl_id = chembl_id or self._gene_to_chembl()

        if not chembl_id:
            return {
                "chemistry_score": 0.5,
                "chemistry_verdict": "UNKNOWN",
                "uniprot_id": uniprot_id,
                "chembl_id": None,
                "warning_signals": ["Could not map protein to a ChEMBL target"],
                "sample_results": [],
            }

        activities = self._fetch_activities(chembl_id)
        result = self._score_activities(activities)
        result["uniprot_id"] = uniprot_id
        result["chembl_id"] = chembl_id
        result["activity_count"] = len(activities)
        return result
