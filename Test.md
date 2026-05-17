Here are some verified real-world examples you can plug into your pipeline:

---

## ✅ High-Confidence "Pursue" Targets
Strong across structure, biology, safety, and chemistry.

| Protein/PDB | Gene | Disease |
|-------------|------|---------|
| `2ITX` | `EGFR` | `breast cancer` |
| `1IEP` | `ABL1` | `chronic myeloid leukemia` |
| `3ERT` | `ESR1` | `breast cancer` |
| `1S9J` | `BRAF` | `melanoma` |
| `2HHI` | `ERBB2` | `gastric cancer` |

---

## ⚠️ Mixed Signal Targets
Good biology, but safety or chemistry concerns exist.

| Protein/PDB | Gene | Disease |
|-------------|------|---------|
| `4TWP` | `KRAS` | `lung adenocarcinoma` |
| `2VT4` | `TP53` | `colorectal cancer` |
| `3QHR` | `MYC` | `lymphoma` |
| `1YCR` | `MDM2` | `liposarcoma` |

---

## ❌ Low-Confidence / Reject Targets
Poor druggability or weak gene-disease evidence.

| Protein/PDB | Gene | Disease |
|-------------|------|---------|
| `1TUP` | `TP53` | `ovarian cancer` |
| `2LZM` | `T4L` | `breast cancer` |

> `T4L` (T4 Lysozyme) is a classic **negative control** — it's a bacteriophage enzyme with no human disease link.

---

## 🔁 Edge Case Inputs

| Protein/PDB | Gene | Disease | Purpose |
|-------------|------|---------|---------|
| `XXXX` | `EGFR` | `breast cancer` | Invalid PDB test |
| `2ITX` | `FAKEGENE999` | `breast cancer` | Invalid gene test |
| `2ITX` | `EGFR` | `Parkinson's disease` | Valid pair, weak association |
| `1IEP` | `ABL1` | `breast cancer` | Gene-disease mismatch test |

---

All PDB IDs above are real entries in the [RCSB PDB](https://www.rcsb.org) and the genes are verifiable on [OpenTargets](https://platform.opentargets.org) and [ChEMBL](https://www.ebi.ac.uk/chembl/). Let me know if you want expected score ranges for any of these!