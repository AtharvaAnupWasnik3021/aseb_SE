# ProteinAI backend

Run the integrated decision API from this directory:

```powershell
python simple_server.py
```

The frontend expects:

- `GET /health`
- `POST /analyze`

Example JSON body:

```json
{
  "protein_input": "2ITX",
  "gene_symbol": "EGFR",
  "disease_name": "breast cancer"
}
```
