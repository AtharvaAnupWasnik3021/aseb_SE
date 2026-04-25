import requests

OPEN_TARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"


# -------------------------------
# GET ENSEMBL ID
# -------------------------------
def get_ensembl_id(gene_symbol: str):
    query = """
    query SearchTarget($symbol: String!) {
      search(queryString: $symbol, entityNames: ["target"]) {
        hits {
          object {
            ... on Target {
              id
              approvedSymbol
              approvedName
            }
          }
        }
      }
    }
    """

    try:
        res = requests.post(
            OPEN_TARGETS_URL,
            json={"query": query, "variables": {"symbol": gene_symbol}},
            timeout=10
        )
        data = res.json()
        hits = data.get("data", {}).get("search", {}).get("hits", [])

        for hit in hits:
            obj = hit.get("object", {})
            if obj.get("approvedSymbol", "").upper() == gene_symbol.upper():
                return obj["id"], obj["approvedName"]

        if hits:
            obj = hits[0]["object"]
            return obj.get("id"), obj.get("approvedName")

    except Exception:
        pass

    return None, None


# -------------------------------
# GET ASSOCIATIONS
# -------------------------------
def get_disease_associations(ensembl_id: str):
    query = """
    query GetAssociations($id: String!) {
      target(ensemblId: $id) {
        associatedDiseases(page: {index: 0, size: 200}) {
          rows {
            score
            disease { name }
            datatypeScores { id score }
          }
        }
      }
    }
    """

    try:
        res = requests.post(
            OPEN_TARGETS_URL,
            json={"query": query, "variables": {"id": ensembl_id}},
            timeout=10
        )
        data = res.json()
        return data.get("data", {}).get("target", {}) \
                   .get("associatedDiseases", {}).get("rows", [])
    except Exception:
        return []


# -------------------------------
# FIXED MATCHING
# -------------------------------
def find_disease_match(associations: list, disease_query: str):
    disease_query = disease_query.lower().strip()

    # Aliases: map common terms to what Open Targets actually calls them
    aliases = {
        "hiv infection": ["hiv", "human immunodeficiency", "acquired immunodeficiency", "aids"],
        "lung cancer": ["lung carcinoma", "lung adenocarcinoma", "non-small cell lung"],
        "breast cancer": ["breast carcinoma", "breast adenocarcinoma"],
    }

    search_terms = [disease_query] + aliases.get(disease_query, disease_query.split())

    best_match = None
    best_score = 0

    for row in associations:
        name = row["disease"]["name"].lower()
        if any(term in name for term in search_terms):
            if row["score"] > best_score:
                best_match = row
                best_score = row["score"]

    if not best_match and associations:
        best_match = max(associations, key=lambda x: x["score"])

    return best_match


# -------------------------------
# EVIDENCE PARSER
# -------------------------------
def parse_evidence_types(datatype_scores: list):
    mapping = {
        "genetic_association": "Genetic mutations in patients",
        "somatic_mutation": "Somatic mutations in disease tissue",
        "known_drug": "Existing drugs target this protein",
        "affected_pathway": "Protein is in disease pathway",
        "literature": "Mentioned in research papers",
        "animal_model": "Animal studies",
        "rna_expression": "Expression changes",
        "text_mining": "Text mining",
        "clinical":            "Clinical trial evidence",        # ← add
        "genetic_literature":  "Genetic evidence from literature"
    }

    parsed = {}
    for item in datatype_scores:
        label = mapping.get(item["id"], item["id"])
        parsed[label] = round(item["score"], 3)

    return parsed


# -------------------------------
# SCORING (FIXED)
# -------------------------------
def rule_based_assessment(overall_score: float, evidence: dict, disease_name: str):

    flags = []
    warnings = []

    disease_lower = disease_name.lower()
    is_pathogen = any(x in disease_lower for x in ["virus", "hiv", "bacteria"])

    has_drug    = evidence.get("Existing drugs target this protein", 0) > 0.2
    clinical    = evidence.get("Clinical trial evidence", 0)
    genetic     = evidence.get("Genetic mutations in patients", 0)
    somatic     = evidence.get("Somatic mutations in disease tissue", 0)
    pathway     = evidence.get("Protein is in disease pathway", 0)

    # Boost 1: known approved drug
    if has_drug:
        overall_score = max(overall_score, 0.65)
        flags.append("Validated drug target (existing drugs)")

    # Boost 2: strong clinical + strong somatic (catches amplification-driven targets like ERBB2)
    if clinical > 0.5:
        flags.append("Strong clinical trial evidence")
        if overall_score >= 0.6 and (has_drug or somatic > 0.4):
            overall_score = max(overall_score, 0.75)

    if not is_pathogen:
        if genetic > 0.3:
            flags.append("Strong genetic evidence")
        if somatic > 0.4:
            flags.append("Strong somatic mutation evidence")
        if genetic < 0.1 and somatic < 0.1:
            warnings.append("Weak genetic support")

    if pathway > 0.2:
        flags.append("Involved in disease pathway")

    if overall_score >= 0.7:
        verdict = "STRONG TARGET"
    elif overall_score >= 0.5:
        verdict = "MODERATE TARGET"
    elif overall_score >= 0.3:
        verdict = "WEAK TARGET"
    else:
        verdict = "AVOID - insufficient disease link"

    return {
        "overall_score": round(overall_score, 3),
        "verdict": verdict,
        "positive_signals": flags,
        "warning_signals": warnings
    }


# -------------------------------
# MAIN FUNCTION (COMPATIBLE)
# -------------------------------
def biology_agent(gene_symbol: str, disease_name: str):

    ensembl_id, full_name = get_ensembl_id(gene_symbol)

    if not ensembl_id:
        return None

    associations = get_disease_associations(ensembl_id)

    if not associations:
        return None

    match = find_disease_match(associations, disease_name)

    if not match:
        return None

    overall_score = match.get("score", 0)
    evidence = parse_evidence_types(match.get("datatypeScores", []))

    assessment = rule_based_assessment(overall_score, evidence, disease_name)

    return {
        "gene_symbol": gene_symbol,
        "disease": match["disease"]["name"],
        "overall_score": assessment["overall_score"],
        "verdict": assessment["verdict"],
        "positive_signals": assessment["positive_signals"],
        "warning_signals": assessment["warning_signals"],
        "evidence": evidence
    }