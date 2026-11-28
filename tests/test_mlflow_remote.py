"""Checks the fail-fast MLflow reachability helper with a throw-away local HTTP server."""

import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.mlflow_remote import ensure_server_reachable, is_remote


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *args):
        pass


class MlflowRemoteTests(unittest.TestCase):
    def test_local_uris_are_not_checked(self):
        self.assertFalse(is_remote("sqlite:///x.db"))
        self.assertTrue(ensure_server_reachable("sqlite:///x.db"))
        self.assertTrue(ensure_server_reachable(None))

    def test_healthy_server_passes(self):
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            self.assertTrue(ensure_server_reachable(f"http://127.0.0.1:{server.server_port}/"))
        finally:
            server.shutdown()
            server.server_close()

    def test_unreachable_server_gives_clear_error(self):
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        port = server.server_port
        server.server_close()                              # port is now closed
        with self.assertRaises(RuntimeError) as ctx:
            ensure_server_reachable(f"http://127.0.0.1:{port}", timeout=2)
        self.assertIn("not reachable", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
