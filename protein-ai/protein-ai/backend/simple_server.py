import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from decision_agent import DecisionAgent


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "healthy"})
            return
        if self.path == "/visualization":
            visualization_path = Path("final_visualization.html")
            if not visualization_path.exists():
                self._send_json(404, {"detail": "Visualization not found"})
                return
            body = visualization_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_json(404, {"detail": "Not found"})

    def do_POST(self):
        if self.path != "/analyze":
            self._send_json(404, {"detail": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = DecisionAgent(
                protein_input=payload["protein_input"].strip(),
                gene_symbol=payload["gene_symbol"].strip(),
                disease_name=payload["disease_name"].strip(),
            ).run()
            self._send_json(200, result)
        except KeyError as exc:
            self._send_json(400, {"detail": f"Missing field: {exc.args[0]}"})
        except Exception as exc:
            self._send_json(500, {"detail": str(exc)})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("ProteinAI backend listening on http://127.0.0.1:8000")
    server.serve_forever()
