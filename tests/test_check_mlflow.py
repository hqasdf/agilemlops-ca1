"""Checks the read-only MLflow preflight script against a throw-away local fake server."""

import contextlib
import io
import json
import threading
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

from scripts import check_mlflow

MLMODEL = b"artifact_path: model\nflavors:\n  python_function:\n    loader_module: mlflow.pyfunc\n"


class _State:
    def __init__(self):
        self.requests = []
        self.aliases = {"champion": ("1", "READY", "run-champion"), "runner-up": ("2", "READY", "run-runner")}
        self.run_status = "FINISHED"
        self.model_files = [{"path": "model/MLmodel", "is_dir": False}]
        self.run_names = ["rf", "cat", "cat"]


def _handler(state):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.end_headers()
            self.wfile.write(data)

        def _route(self, body):
            parts = urlsplit(self.path)
            query = {key: values[0] for key, values in parse_qs(parts.query).items()}
            state.requests.append((self.command, parts.path))
            if parts.path == "/health":
                return self._send(200, b"OK", "text/plain")
            if parts.path == "/api/2.0/mlflow/experiments/search":
                return self._send(200, {"experiments": [{"experiment_id": "1", "name": "PBI_test"}]})
            if parts.path == "/api/2.0/mlflow/runs/search":
                runs = [{"info": {"run_name": name}} for name in state.run_names]
                return self._send(200, {"runs": runs})
            if parts.path == "/api/2.0/mlflow/registered-models/alias":
                if query["alias"] not in state.aliases:
                    return self._send(404, {"error_code": "RESOURCE_DOES_NOT_EXIST"})
                version, status, run_id = state.aliases[query["alias"]]
                return self._send(200, {"model_version": {"version": version, "status": status, "run_id": run_id}})
            if parts.path == "/api/2.0/mlflow/runs/get":
                return self._send(200, {"run": {"info": {"status": state.run_status}}})
            if parts.path == "/api/2.0/mlflow/artifacts/list":
                return self._send(200, {"files": state.model_files})
            if parts.path == "/get-artifact":
                return self._send(200, MLMODEL, "text/plain")
            return self._send(404, {})

        def do_GET(self):
            self._route(b"")

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self._route(b"")

        def log_message(self, *args):
            pass

    return Handler


class CheckMlflowTests(unittest.TestCase):
    def setUp(self):
        self.state = _State()
        self.server = HTTPServer(("127.0.0.1", 0), _handler(self.state))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.uri = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def run_main(self, *args):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = check_mlflow.main(list(args))
        return code, buffer.getvalue()

    def test_healthy_server_passes_and_prints_links(self):
        code, output = self.run_main(self.uri, "--timeout", "5")
        self.assertEqual(code, 0, output)
        self.assertIn("PREFLIGHT PASSED", output)
        self.assertIn("experiment 1 PBI_test: 3 runs", output)
        self.assertIn("fraud-detector@champion is version 1, READY", output)
        self.assertIn("#/models/fraud-detector", output)

    def test_repeated_run_names_are_noted_but_do_not_fail(self):
        code, output = self.run_main(self.uri)
        self.assertEqual(code, 0)
        self.assertIn("run names used more than once: cat", output)

    def test_only_read_requests_are_sent(self):
        self.run_main(self.uri)
        paths = {path for _, path in self.state.requests}
        self.assertTrue(paths)
        for method, path in self.state.requests:
            self.assertIn(method, {"GET", "POST"})
            if method == "POST":
                self.assertTrue(path.endswith("/search"), path)
        for word in ("create", "delete", "update", "set-", "log-", "transition", "restore"):
            self.assertFalse(any(word in path for path in paths), word)

    def test_missing_alias_fails(self):
        del self.state.aliases["runner-up"]
        code, output = self.run_main(self.uri)
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] fraud-detector@runner-up is missing", output)

    def test_version_not_ready_fails(self):
        self.state.aliases["champion"] = ("1", "PENDING_REGISTRATION", "run-champion")
        code, output = self.run_main(self.uri)
        self.assertEqual(code, 1)
        self.assertIn("status is PENDING_REGISTRATION", output)

    def test_unfinished_source_run_fails(self):
        self.state.run_status = "FAILED"
        code, output = self.run_main(self.uri)
        self.assertEqual(code, 1)
        self.assertIn("is FAILED, not FINISHED", output)

    def test_missing_model_file_fails(self):
        self.state.model_files = []
        code, output = self.run_main(self.uri)
        self.assertEqual(code, 1)
        self.assertIn("has no model/MLmodel file", output)

    def test_unreachable_server_prints_recovery_steps(self):
        server = HTTPServer(("127.0.0.1", 0), _handler(self.state))
        port = server.server_port
        server.server_close()
        code, output = self.run_main(f"http://127.0.0.1:{port}", "--timeout", "2")
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] /health did not answer", output)
        self.assertIn("Reboot instance", output)
        self.assertIn("PREFLIGHT FAILED", output)

    def test_missing_url_is_a_usage_error(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with unittest.mock.patch.dict("os.environ", {}, clear=True):
                self.assertEqual(check_mlflow.main([]), 2)


if __name__ == "__main__":
    unittest.main()
