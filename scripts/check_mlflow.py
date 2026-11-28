"""Read-only preflight check for the shared MLflow server.

Run it before a demo, before training, and after every AWS Learner Lab restart:

    python -m scripts.check_mlflow                      # uses MLFLOW_TRACKING_URI
    python -m scripts.check_mlflow http://<server-url>  # or pass the URL

It only reads. It never creates experiments, runs, models or aliases, so it cannot change
any result on the server. It checks that:

1. the server answers /health,
2. every experiment lists its runs (and notes run names that were used more than once),
3. the registered model has the champion and runner-up aliases on versions that are READY,
4. each aliased version's source run is FINISHED and its model file can be downloaded
   through the server, which proves the S3 artifact path works and not only the database.

Exit code 0 means everything passed, 1 means a check failed, 2 means the URL is missing.
Only the Python standard library is used, so it runs on any computer.
"""
import argparse
import json
import os
import sys
from collections import Counter
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

DEFAULT_MODEL = "fraud-detector"
ALIASES = ("champion", "runner-up")

RECOVERY = """
The server did not answer. Work through these steps in order:
  1. AWS Academy page: the lab must show a green dot next to AWS.
  2. Elastic Beanstalk, environment mlflow-pbi12. The Warning "Access denied while accessing
     Auto Scaling using role LabRole" is a known Learner Lab message and can be ignored.
  3. "Connection refused": the instance started without installing the app. Restart app
     servers and Deploy did not help on 26 November. Rebooting the instance did:
     EC2 > Instances > mlflow-pbi12 > Instance state > Reboot instance. Wait 2 minutes.
  4. Run this check again.
Do not Rebuild or Terminate the environment. The data is in RDS and S3 and survives a restart.
""".strip("\n")


def _url(base, path, params=None):
    return base + path + ("?" + urlencode(params) if params else "")


def _get(base, path, params=None, timeout=20, raw=False):
    with urlopen(_url(base, path, params), timeout=timeout) as response:
        body = response.read()
    return body if raw else json.loads(body)


def _post(base, path, body, timeout=30):
    request = Request(_url(base, path), json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _reason(error):
    if isinstance(error, HTTPError):
        return f"HTTP {error.code} {error.reason}"
    return str(getattr(error, "reason", error))


def check_health(base, timeout):
    try:
        body = _get(base, "/health", timeout=timeout, raw=True)
    except HTTPError as error:
        return False, (f"/health answered {_reason(error)}. The server or a firewall in front of "
                       "this computer is refusing the request")
    except (URLError, OSError) as error:
        return False, f"/health did not answer ({_reason(error)})"
    return True, f"/health answered {body.decode(errors='replace').strip()[:20] or 'OK'}"


def _pages(base, path, body, key, token_key="next_page_token"):
    """Yield every item of a paged MLflow search call."""
    request = dict(body)
    while True:
        page = _post(base, path, request)
        yield from page.get(key, [])
        token = page.get(token_key)
        if not token:
            return
        request["page_token"] = token


def list_experiments(base):
    """Return (experiment_id, name, run_count, repeated_names) for each active experiment."""
    rows = []
    for experiment in _pages(base, "/api/2.0/mlflow/experiments/search", {"max_results": 1000}, "experiments"):
        names = Counter(
            run["info"].get("run_name", "")
            for run in _pages(base, "/api/2.0/mlflow/runs/search",
                              {"experiment_ids": [experiment["experiment_id"]], "max_results": 1000}, "runs"))
        repeated = sorted(name for name, count in names.items() if count > 1 and name)
        rows.append((experiment["experiment_id"], experiment["name"], sum(names.values()), repeated))
    return rows


def check_alias(base, model, alias, timeout):
    """Check one alias end to end. Returns (ok, message)."""
    label = f"{model}@{alias}"
    try:
        version = _get(base, "/api/2.0/mlflow/registered-models/alias",
                       {"name": model, "alias": alias}, timeout)["model_version"]
    except (HTTPError, URLError, OSError, KeyError) as error:
        return False, f"{label} is missing ({_reason(error) if not isinstance(error, KeyError) else 'no version'})"
    if version.get("status") != "READY":
        return False, f"{label} is version {version.get('version')} but its status is {version.get('status')}"
    run_id = version.get("run_id")
    try:
        run = _get(base, "/api/2.0/mlflow/runs/get", {"run_id": run_id}, timeout)["run"]["info"]
        if run.get("status") != "FINISHED":
            return False, f"{label} source run {run_id[:8]} is {run.get('status')}, not FINISHED"
        files = _get(base, "/api/2.0/mlflow/artifacts/list", {"run_id": run_id, "path": "model"}, timeout)
        if not any(item["path"].endswith("MLmodel") for item in files.get("files", [])):
            return False, f"{label} source run {run_id[:8]} has no model/MLmodel file"
        text = _get(base, "/get-artifact", {"run_id": run_id, "path": "model/MLmodel"}, timeout, raw=True)
        if b"flavors" not in text:
            return False, f"{label} model/MLmodel downloaded but looks wrong"
    except (HTTPError, URLError, OSError, KeyError) as error:
        return False, f"{label} model file could not be read from the server ({_reason(error)})"
    return True, f"{label} is version {version.get('version')}, READY, model file downloaded from the server"


def run_checks(base, model, timeout):
    """Run every check. Returns a list of (ok, message) and a list of notes."""
    results, notes = [], []
    ok, message = check_health(base, timeout)
    results.append((ok, message))
    if not ok:
        return results, notes
    try:
        experiments = list_experiments(base)
    except (HTTPError, URLError, OSError) as error:
        results.append((False, f"experiments could not be listed ({_reason(error)})"))
        return results, notes
    results.append((bool(experiments), f"{len(experiments)} experiments found"))
    for experiment_id, name, count, repeated in experiments:
        notes.append(f"  experiment {experiment_id} {name}: {count} runs")
        if repeated:
            notes.append(f"    note: run names used more than once: {', '.join(repeated[:3])}"
                         f"{' ...' if len(repeated) > 3 else ''} (told apart by start time)")
    for alias in ALIASES:
        results.append(check_alias(base, model, alias, timeout))
    return results, notes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only preflight check of the shared MLflow server.")
    parser.add_argument("uri", nargs="?", default=os.environ.get("MLFLOW_TRACKING_URI"),
                        help="server URL, default MLFLOW_TRACKING_URI")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"registered model, default {DEFAULT_MODEL}")
    parser.add_argument("--timeout", type=int, default=20, help="seconds to wait for each request")
    args = parser.parse_args(argv)
    parts = urlsplit(args.uri or "")
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        print("Give the shared server URL (http://...) or set MLFLOW_TRACKING_URI.")
        return 2
    base = args.uri.rstrip("/")
    print(f"Preflight check of {base} (read-only)")
    results, notes = run_checks(base, args.model, args.timeout)
    for ok, message in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {message}")
    for line in notes:
        print(line)
    failed = sum(1 for ok, _ in results if not ok)
    if not results[0][0]:
        print(RECOVERY)
    if failed:
        print(f"PREFLIGHT FAILED: {failed} check(s) did not pass.")
        return 1
    print("PREFLIGHT PASSED")
    print(f"Open: {base}/#/experiments   {base}/#/models/{args.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
