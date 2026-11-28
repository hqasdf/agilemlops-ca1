"""Publish and verify the approved PBI 06 run tree without fitting models."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from time import sleep
from urllib.parse import urlsplit

import joblib
import mlflow
from mlflow.entities import Param
from mlflow.exceptions import MlflowException
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

from src.feature_engineering import INPUT_FEATURES
from src.pbi06_research import EXPERIMENT
from src.pbi06_tracking import COMPLETED_RUN, LEGACY_EXPERIMENT, LOCAL_URI, rename_rebuild_experiment

SOURCE = "pbi06.source_run_id"
TREE = "pbi06.source_parent_id"
PARENT = "mlflow.parentRunId"
VERIFIED = "pbi06.verified_payload"
EXPECTED_RUNS = 491
PUBLISH_DIR = Path("artifacts/pbi06_publish")
RECOVERY_TAG = "pbi06.recovery_source"
EXPECTED_STAGES = {"baseline": 21, "ordinary": 133, "advanced": 329, "optimized": 7}
APPROVED_FILES = {
    "configuration.json", "parameters.json", "model/model.joblib",
    "model/preprocessor.joblib", "model/inference.json", "evidence.json",
    "search_spaces.json", "stages/baseline.json", "stages/ordinary.json",
    "stages/advanced.json", "folds/1/preprocessor.joblib", "folds/2/preprocessor.joblib",
}
SENSITIVE = re.compile(
    r'(?i)AKIA[A-Z0-9]{16}|ASIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----'
    r'|https?://[^/@\s]+:[^/@\s]+@'
    r'|["\x27](customer_id|order_id|password|secret_access_key|access_token)["\x27]\s*:'
)


def _hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _safe_text(text, context):
    if SENSITIVE.search(text):
        raise ValueError(f"Potential individual identifier or credential in {context}")


def _shared_client(uri):
    parts = urlsplit(uri or "")
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("Set MLFLOW_TRACKING_URI to the approved shared HTTP(S) endpoint")
    return mlflow.MlflowClient(tracking_uri=uri)


def _list_runs(client, experiment_id, query):
    token = None
    while True:
        page = client.search_runs([experiment_id], filter_string=query, max_results=1000, page_token=token)
        yield from page
        token = page.token
        if not token:
            break


def _files(client, run_id, path=None):
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from _files(client, run_id, item.path)
        else:
            yield item.path


def _source_tree(client):
    exp = client.get_experiment_by_name(EXPERIMENT)
    root = client.get_run(COMPLETED_RUN)
    children = list(_list_runs(client, exp.experiment_id, f"tags.`{PARENT}` = '{COMPLETED_RUN}'"))
    stages = Counter(run.data.tags.get("stage") for run in children)
    if (root.info.experiment_id != exp.experiment_id or root.data.tags.get("verified_complete") != "true"
            or stages != EXPECTED_STAGES or len(children) + 1 != EXPECTED_RUNS
            or any(run.info.status != "FINISHED" for run in [root, *children])):
        raise RuntimeError("Local research tree does not match the approved publication scope")
    return exp, root, children


def _export_reports(root, *, frozen=False):
    reports = Path(__file__).resolve().parents[1] / "reports"
    exports = {}
    for name in ("PBI06_model_comparison.md", "PBI06_training_rebuild_report.md"):
        dest = root / name
        if frozen and dest.is_file():
            _safe_text(dest.read_text(), name)
            exports[f"reports/{name}"] = dest
            continue
        text = (reports / name).read_text()
        if name == "PBI06_model_comparison.md":
            text = text.partition("<details>")[0].rstrip() + "\n"
        if COMPLETED_RUN not in text:
            raise ValueError(f"{name} does not identify the approved source run")
        _safe_text(text, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        exports[f"reports/{name}"] = dest
    return exports


def _package(client, runs, cache, report_exports):
    files, excluded = {}, []
    for run in runs:
        rid = run.info.run_id
        _safe_text(json.dumps([run.data.params, run.data.tags]), f"run {rid} metadata")
        artifacts = {}
        for name in _files(client, rid):
            if rid == COMPLETED_RUN and name == "predictions.npz":
                excluded.append({"path": name, "bytes": Path(client.download_artifacts(rid, name)).stat().st_size})
                continue
            if name not in APPROVED_FILES:
                raise ValueError(f"Unreviewed artifact path {name} on {rid}")
            path = Path(client.download_artifacts(rid, name))
            if name.endswith(".json"):
                _safe_text(path.read_text(), f"{rid}/{name}")
            if name == "model/model.joblib" and run.data.tags.get("family") == "random_forest":
                sanitized = cache / rid / _hash(path) / "model.joblib"
                if not sanitized.is_file():
                    model = joblib.load(path)
                    if not isinstance(model, RandomForestClassifier):
                        raise ValueError(f"Unexpected forest artifact type on {rid}")
                    probe = np.zeros((2, model.n_features_in_), dtype=np.float32)
                    probe[1] = 1.0
                    scores = model.predict_proba(probe)
                    model._sample_weight = None
                    np.testing.assert_allclose(model.predict_proba(probe), scores, rtol=1e-12, atol=1e-14)
                    sanitized.parent.mkdir(parents=True, exist_ok=True)
                    partial = sanitized.with_suffix(".pending")
                    joblib.dump(model, partial, compress=3)
                    partial.replace(sanitized)
                path = sanitized
            artifacts[name] = path
        files[rid] = artifacts
    files[COMPLETED_RUN].update(_export_reports(report_exports, frozen=True))
    return files, excluded


def _metric_key(metric):
    return (
        metric.key, metric.timestamp, metric.step, float(metric.value).hex(),
        metric.dataset_name, metric.dataset_digest, metric.model_id, metric.run_id,
    )


def _history(client, run):
    return [m for k in sorted(run.data.metrics) for m in client.get_metric_history(run.info.run_id, k)]


def _hist_keys(history):
    # Tracking can populate the metric's own run_id on read. Source event fields
    # are identical apart from that destination-assigned ID.
    return set(_metric_key(metric)[:-1] for metric in history)


CHUNK_BYTES = 768 * 1024


def _chunk_dir(name):
    return f"{name}.parts"


def _upload_chunks(client, run_id, name, source, temp):
    """Use proxy-compatible uploads after an HTTP 413 response."""
    root = Path(temp) / "upload_chunks"
    root.mkdir(exist_ok=True)
    parts = []
    with source.open("rb") as stream:
        while block := stream.read(CHUNK_BYTES):
            filename = f"{len(parts):04}.part"
            local = root / filename
            local.write_bytes(block)
            client.log_artifact(run_id, str(local), _chunk_dir(name))
            parts.append({"name": filename, "sha256": _hash(local)})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"sha256": _hash(source), "bytes": source.stat().st_size,
                                   "parts": parts}, sort_keys=True) + "\n")
    client.log_artifact(run_id, str(manifest), _chunk_dir(name))


def log_shared_artifact(client, run_id, name, source, temp):
    """Log a file through the MLflow proxy, splitting it only if required."""
    artifact_dir = str(Path(name).parent)
    try:
        client.log_artifact(run_id, str(source), None if artifact_dir == "." else artifact_dir)
    except Exception as error:
        if "413" not in str(error) or source.stat().st_size <= CHUNK_BYTES:
            raise
        _upload_chunks(client, run_id, name, source, temp)


def download_published_artifact(client, run_id, path, directory, *, split_info=None):
    """Download a standard artifact or reconstruct a proxy-sized split artifact."""
    directory = Path(directory)
    siblings = {i.path for i in client.list_artifacts(run_id, str(Path(path).parent))
                } if "/" in path else {i.path for i in client.list_artifacts(run_id)}
    if path in siblings:
        return Path(client.download_artifacts(run_id, path, dst_path=str(directory)))
    prefix = _chunk_dir(path)
    manifest_path = Path(client.download_artifacts(run_id, f"{prefix}/manifest.json", dst_path=str(directory)))
    manifest = json.loads(manifest_path.read_text())
    if split_info is not None:
        split_info[path] = {"parts": len(manifest["parts"]), "manifest_bytes": manifest_path.stat().st_size}
    result = directory / path
    result.parent.mkdir(parents=True, exist_ok=True)
    with result.open("wb") as assembled:
        for i, part in enumerate(manifest["parts"]):
            if part["name"] != f"{i:04}.part":
                raise RuntimeError(f"Unexpected part name in {prefix}")
            local = Path(client.download_artifacts(run_id, f"{prefix}/{part['name']}", dst_path=str(directory)))
            if _hash(local) != part["sha256"]:
                raise RuntimeError(f"Part checksum mismatch for {run_id}/{prefix}/{part['name']}")
            with local.open("rb") as handle:
                assembled.write(handle.read())
    if _hash(result) != manifest["sha256"] or result.stat().st_size != manifest["bytes"]:
        raise RuntimeError(f"Reassembled checksum mismatch for {run_id}/{path}")
    return result


def _same_sanitized_forest(source, uploaded):
    """Validate tree state for serialized copies produced by an earlier attempt."""
    local, remote = joblib.load(source), joblib.load(uploaded)
    if (not isinstance(local, RandomForestClassifier) or not isinstance(remote, RandomForestClassifier)
            or local._sample_weight is not None or remote._sample_weight is not None
            or local.get_params() != remote.get_params()
            or len(local.estimators_) != len(remote.estimators_)):
        raise RuntimeError("Shared forest does not match the approved sanitized estimator")
    np.testing.assert_array_equal(local.classes_, remote.classes_)
    if local.n_features_in_ != remote.n_features_in_:
        raise RuntimeError("Shared forest feature count differs")
    for left, right in zip(local.estimators_, remote.estimators_):
        a, b = left.tree_.__getstate__(), right.tree_.__getstate__()
        if a["max_depth"] != b["max_depth"] or a["node_count"] != b["node_count"]:
            raise RuntimeError("Shared forest tree dimensions differ")
        np.testing.assert_array_equal(a["nodes"], b["nodes"])
        np.testing.assert_array_equal(a["values"], b["values"])


def _add_files(client, run_id, files, temp, *, sanitized_forest=False):
    existing = set(_files(client, run_id))
    if unexpected := {p for p in existing if p not in files and
                      not any(p.startswith(f"{_chunk_dir(name)}/") for name in files)}:
        raise RuntimeError(f"Unexpected shared artifacts for {run_id}: {unexpected}")
    split = {}
    for name, source in files.items():
        prefix = _chunk_dir(name)
        if name in existing or f"{prefix}/manifest.json" in existing:
            uploaded = download_published_artifact(client, run_id, name, temp, split_info=split)
            if _hash(uploaded) != _hash(source):
                if not sanitized_forest or name != "model/model.joblib":
                    raise RuntimeError(f"Existing shared checksum differs for {run_id}/{name}")
                _same_sanitized_forest(source, uploaded)
                # Preserve a semantically identical previously uploaded copy.
                partial = source.with_suffix(".pending")
                shutil.copyfile(uploaded, partial)
                partial.replace(source)
            continue
        if any(p.startswith(f"{prefix}/") for p in existing):
            _upload_chunks(client, run_id, name, source, temp)
        else:
            log_shared_artifact(client, run_id, name, source, temp)
        if _hash(download_published_artifact(client, run_id, name, temp, split_info=split)) != _hash(source):
            raise RuntimeError(f"Shared checksum mismatch for {run_id}/{name}")

    return split


def _existing_copies(client, exp_id):
    result = {}
    for run in _list_runs(client, exp_id, f"tags.`{TREE}` = '{COMPLETED_RUN}'"):
        original = run.data.tags.get(SOURCE)
        if not original or original in result:
            raise RuntimeError(f"Duplicate or missing source reference in AWS: {original}")
        result[original] = run
    return result


def _create_copy(client, exp_id, copies, source, parent_id):
    rid = source.info.run_id
    tags = {key: value for key, value in source.data.tags.items() if key != PARENT}
    tags.update({SOURCE: rid, TREE: COMPLETED_RUN})
    if parent_id:
        tags[PARENT] = parent_id
    current = copies.get(rid)
    if current is None:
        return client.create_run(exp_id, start_time=source.info.start_time,
                                 tags=tags, run_name=source.info.run_name), True
    if (current.info.start_time != source.info.start_time
            or current.info.run_name != source.info.run_name
            or current.data.tags.get(PARENT) != parent_id
            or any(current.data.tags.get(k) != v for k, v in tags.items())):
        raise RuntimeError(f"Shared run metadata conflict for {rid}")
    return current, False


def _sync_run(source_client, target, local, remote, files):
    shared_id = remote.info.run_id
    def payload_hash():
        manifest = {name: _hash(file) for name, file in files.items()}
        return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

    payload = payload_hash()
    if (remote.data.tags.get(VERIFIED) == payload and remote.info.status == local.info.status
            and remote.info.end_time == local.info.end_time and remote.data.params == local.data.params
            and remote.data.metrics == local.data.metrics):
        return False
    if any(local.data.params.get(key) != val for key, val in remote.data.params.items()):
        raise RuntimeError(f"Shared parameter conflict on {shared_id}")
    history = _history(source_client, local)
    previous = _hist_keys(_history(target, remote))
    wanted = _hist_keys(history)
    if not previous.issubset(wanted):
        raise RuntimeError(f"Unexpected metric events on {shared_id}")
    missing = [m for m in history if _metric_key(m)[:-1] not in previous]
    params = [Param(k, v) for k, v in local.data.params.items() if k not in remote.data.params]
    for pos in range(0, max(len(missing), 3 * len(params)), 300):
        target.log_batch(shared_id, metrics=missing[pos:pos + 300],
                         params=params[pos // 3:pos // 3 + 100], synchronous=True)
    current = target.get_run(shared_id)
    if current.data.params != local.data.params or current.data.metrics != local.data.metrics:
        raise RuntimeError(f"Shared parameters or latest metrics differ on {shared_id}")
    if _hist_keys(_history(target, current)) != wanted:
        raise RuntimeError(f"Shared metric history differs on {shared_id}")
    with TemporaryDirectory() as tmp:
        split = _add_files(target, shared_id, files, tmp,
                           sanitized_forest=local.data.tags.get("family") == "random_forest")
    if split:
        target.set_tag(shared_id, "pbi06.split_files", json.dumps(split, sort_keys=True))
    target.set_terminated(shared_id, status=local.info.status, end_time=local.info.end_time)
    target.set_tag(shared_id, VERIFIED, payload_hash())
    return True


def _verify_final(target, evidence, run_map):
    result = {}
    for row in evidence["final"]:
        shared = run_map[row["run_id"]]
        with TemporaryDirectory() as tmp:
            paths = {name: download_published_artifact(target, shared, f"model/{name}", tmp)
                     for name in ("model.joblib", "preprocessor.joblib", "inference.json")}
            model = joblib.load(paths["model.joblib"])
            preprocessor = joblib.load(paths["preprocessor.joblib"])
            metadata = json.loads(paths["inference.json"].read_text())
            if (metadata["threshold"] != row["threshold"]
                    or metadata["input_features"] != INPUT_FEATURES
                    or list(preprocessor.feature_names_in_) != INPUT_FEATURES
                    or list(model.classes_) != [0, 1]
                    or not callable(model.predict_proba)):
                raise RuntimeError(f"Invalid final shared artifact on {shared}")
            if isinstance(model, RandomForestClassifier) and model._sample_weight is not None:
                raise RuntimeError("Shared forest retained its per-row training weights")
        result[row["family"]] = shared
    return result


def _recorded_destination():
    path = PUBLISH_DIR / "destination.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _check_destination(uri, experiment, original, copies=None):
    if original is None:
        return
    if (uri.rstrip("/") != original["tracking_uri"]
            or experiment is None or experiment.experiment_id != original["experiment_id"]):
        raise RuntimeError("Recorded AWS experiment is missing or the destination changed. "
                           "Resume against the original shared backend.")
    if copies is not None:
        parent = copies.get(COMPLETED_RUN)
        if parent is None or parent.info.run_id != original["parent_run_id"]:
            raise RuntimeError("Recorded AWS parent is missing or changed. "
                               "Resume against the original shared backend.")


def shared_status(uri, source_run=COMPLETED_RUN):
    """Read shared publication state without creating runs or changing logging."""
    client = _shared_client(uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        previous = _recorded_destination()
        if previous and uri.rstrip("/") == previous["tracking_uri"]:
            return {
                "status": "Shared server no longer lists the recorded experiment",
                "experiment": EXPERIMENT, "previous": previous,
            }
        return {"status": "Not published", "experiment": EXPERIMENT}
    copies = _existing_copies(client, experiment.experiment_id) if source_run == COMPLETED_RUN else {}
    parent = copies.get(source_run)
    if parent is None:
        # Future full research runs can originate on the shared server.
        try:
            candidate = client.get_run(source_run)
        except MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
        else:
            if candidate.info.experiment_id == experiment.experiment_id:
                children = _list_runs(client, experiment.experiment_id, f"tags.`{PARENT}` = '{source_run}'")
                final = {r.data.tags["family"]: r.info.run_id for r in children
                         if r.data.tags.get("stage") == "optimized"}
                return {"status": "Tracked directly on AWS", "experiment": EXPERIMENT,
                        "experiment_id": experiment.experiment_id, "parent_run_id": source_run,
                        "final_runs": final, "dashboard": f"{uri.rstrip('/')}/#/experiments/{experiment.experiment_id}"}
    verified = sum(bool(r.data.tags.get(VERIFIED)) for r in copies.values())
    return {
        "status": ("Published and verified" if parent and parent.data.tags.get("pbi06.tree_verified") == "true"
                   and verified == len(copies) == EXPECTED_RUNS else f"{verified}/{EXPECTED_RUNS} runs verified"),
        "experiment": EXPERIMENT, "experiment_id": experiment.experiment_id,
        "parent_run_id": parent.info.run_id if parent else None,
        "final_runs": {r.data.tags["family"]: r.info.run_id for r in copies.values()
                       if r.data.tags.get("stage") == "optimized" and r.data.tags.get(VERIFIED)},
        "dashboard": f"{uri.rstrip('/')}/#/experiments/{experiment.experiment_id}",
    }


def rename_recorded_shared(uri):
    """Rename the recorded AWS experiment without creating a new run."""
    client = _shared_client(uri)
    previous = _recorded_destination()
    if not previous or uri.rstrip("/") != previous["tracking_uri"]:
        raise RuntimeError("Configure the recorded shared backend before renaming.")
    experiment = rename_rebuild_experiment(
        client, previous["experiment_id"], previous["parent_run_id"], shared=True
    )
    if experiment is None:
        return {"status": "Original shared experiment is missing", "experiment": EXPERIMENT}
    return shared_status(uri)


def _publication_target(client, uri, previous, recover_replacement):
    """Identify an existing publication or authorize an explicitly confirmed replacement."""
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if client.get_experiment_by_name(LEGACY_EXPERIMENT):
        raise RuntimeError("The original shared experiment exists. Rename it in place before publishing.")
    copies = _existing_copies(client, experiment.experiment_id) if experiment else {}
    try:
        _check_destination(uri, experiment, previous, copies)
    except RuntimeError:
        if not recover_replacement:
            raise
    else:
        if previous or COMPLETED_RUN in copies:
            return experiment, copies, False
    if not recover_replacement:
        raise RuntimeError("Creating a replacement requires --recover-replacement after backend confirmation.")

    if previous:
        try:
            client.get_run(previous["parent_run_id"])
        except MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
        else:
            raise RuntimeError("The recorded AWS parent still exists. Do not create a replacement.")
    if experiment:
        if experiment.tags.get(RECOVERY_TAG) != COMPLETED_RUN:
            raise RuntimeError("The replacement name is already used by a different experiment.")
        runs = list(_list_runs(client, experiment.experiment_id, ""))
        if {r.info.run_id for r in runs} != {r.info.run_id for r in copies.values()}:
            raise RuntimeError("The replacement experiment contains unrelated runs.")
    return experiment, copies, True


def _write_checkpoint(path, data):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def verify_readiness(client, probe, *, check_artifact=False):
    """Check that the synthetic probe remains on the same tracking and artifact store."""
    run = client.get_run(probe["run_id"])
    if (run.info.experiment_id != "0" or run.info.status != "FINISHED"
            or run.data.params.get("scope") != "synthetic" or run.data.metrics.get("probe") != 1.0):
        raise RuntimeError("The shared readiness run is missing or changed.")
    if check_artifact:
        with TemporaryDirectory() as tmp:
            artifact = Path(client.download_artifacts(run.info.run_id, "readiness.txt", dst_path=tmp))
            if _hash(artifact) != probe["sha256"]:
                raise RuntimeError("The shared readiness artifact is missing or changed.")


def _check_batch(client, exp_id, parent_id, run_map, completed, readiness):
    if readiness:
        verify_readiness(client, readiness)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if not experiment or experiment.experiment_id != exp_id or experiment.lifecycle_stage != "active":
        raise RuntimeError("Shared experiment disappeared or changed between publication batches.")
    remote = _existing_copies(client, exp_id)
    for local_id, shared_id in run_map.items():
        run = remote.get(local_id)
        expected_parent = None if local_id == COMPLETED_RUN else parent_id
        if not run or run.info.run_id != shared_id or run.data.tags.get(PARENT) != expected_parent:
            raise RuntimeError(f"Shared run {local_id} disappeared or changed between batches.")
        if local_id in completed and (run.info.status != "FINISHED" or not run.data.tags.get(VERIFIED)):
            raise RuntimeError(f"Shared run {local_id} lost its verification state.")
    return remote


def _copy_batch(batch, sync, workers):
    """Stop scheduling writes after the first error in a bounded batch."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(sync, run): run.info.run_id for run in batch}
        try:
            return [future.result() for future in as_completed(futures)]
        except Exception:
            for future in futures:
                future.cancel()
            raise


def publish(uri, workers=4, *, recover_replacement=False, batch_size=24, readiness=None):
    """Resume the approved local-to-shared publication and return a verified receipt."""
    target = _shared_client(uri)
    if not 1 <= batch_size <= 48 or not 1 <= workers <= 6:
        raise ValueError("Publication requires batches of 1–48 runs and at most 6 workers.")
    if recover_replacement and readiness is None:
        raise ValueError("Replacement publication requires a verified --readiness-file.")
    if readiness:
        verify_readiness(target, readiness, check_artifact=True)
    source = mlflow.MlflowClient(tracking_uri=LOCAL_URI)
    exp, root, children = _source_tree(source)
    approved = {r.info.run_id for r in [root, *children]}
    workdir = PUBLISH_DIR
    workdir.mkdir(parents=True, exist_ok=True)
    with (workdir / "publish.lock").open("a") as lock, TemporaryDirectory() as tmp, threadpool_limits(limits=4):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = _recorded_destination()
        experiment, copies, replacing = _publication_target(target, uri, previous, recover_replacement)
        report_version = previous.get("reports_version", "current") if previous and not replacing else "current"
        report_exports = workdir / f"{report_version}_report_exports"
        if report_version == "original" and not (report_exports / "PBI06_model_comparison.md").is_file():
            raise RuntimeError("Restore the original archived report exports before resuming this destination.")
        files, excluded = _package(source, [root, *children], workdir / "forests", report_exports)
        if readiness:
            verify_readiness(target, readiness, check_artifact=True)
        if experiment is None:
            exp_id = target.create_experiment(EXPERIMENT, tags={**exp.tags, RECOVERY_TAG: COMPLETED_RUN})
            experiment = target.get_experiment(exp_id)
        exp_id = experiment.experiment_id
        if experiment.lifecycle_stage != "active" or not experiment.artifact_location.startswith("mlflow-artifacts:"):
            raise RuntimeError("Shared experiment is inactive or does not use server-proxied artifacts")
        if any(experiment.tags.get(k) != v for k, v in exp.tags.items()):
            raise RuntimeError("Shared experiment metadata differs from the source")
        if set(copies) - approved:
            raise RuntimeError("Unexpected shared source references")
        parent, new = _create_copy(target, exp_id, copies, root, None)
        parent_id = parent.info.run_id
        checkpoint = workdir / "destination.json"
        destination = {"tracking_uri": uri.rstrip("/"), "experiment_id": exp_id,
                       "parent_run_id": parent_id, "reports_version": report_version}
        if previous and replacing:
            archived = workdir / f"previous_destination_{previous['parent_run_id']}.json"
            _write_checkpoint(archived, previous)
        _write_checkpoint(checkpoint, {**((previous or {}) if not replacing else {}), **destination})
        print(f"Experiment {exp_id}, parent {parent_id}, resuming {len(copies)} existing runs", flush=True)

        def sync(local):
            remote, created = _create_copy(target, exp_id, copies, local, parent_id)
            verified = _sync_run(source, target, local, remote, files[local.info.run_id])
            return local.info.run_id, remote.info.run_id, created, verified

        run_map = {key: run.info.run_id for key, run in copies.items()}
        run_map[COMPLETED_RUN] = parent_id
        created = int(new)
        verified_now = 0
        completed = set()
        evidence = json.loads(files[COMPLETED_RUN]["evidence.json"].read_text())
        final_ids = {row["run_id"] for row in evidence["final"]}
        first = [run for run in children if run.info.run_id in final_ids]
        remaining = [run for run in children if run.info.run_id not in final_ids]
        if len(first) != EXPECTED_STAGES["optimized"]:
            raise RuntimeError("Expected exactly seven selected final models.")
        batches = [first] + [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]
        _check_batch(target, exp_id, parent_id, run_map, completed, readiness)
        for i, batch in enumerate(batches, 1):
            if i > 1:
                sleep(3)
                _check_batch(target, exp_id, parent_id, run_map, completed, readiness)
            try:
                for local_id, remote_id, made, checked in _copy_batch(batch, sync, workers):
                    run_map[local_id] = remote_id
                    created += made
                    verified_now += checked
                    completed.add(local_id)
                _check_batch(target, exp_id, parent_id, run_map, completed, readiness)
                if i == 1:
                    if readiness:
                        verify_readiness(target, readiness, check_artifact=True)
                    destination["final_runs"] = _verify_final(target, evidence, run_map)
                    _write_checkpoint(checkpoint, destination)
                    print(f"Seven final models downloaded and loaded: {destination['final_runs']}", flush=True)
            except Exception as error:
                _write_checkpoint(workdir / "failures.json", {"batch": i, "error": str(error),
                                  "confirmed_child_runs": len(completed), "parent_run_id": parent_id})
                raise RuntimeError(f"Publication stopped after a failure in batch {i}. "
                                   "No later batches were started.") from error
            print(f"Batch {i}/{len(batches)} confirmed, {len(completed)}/{len(children)} children", flush=True)
        map_path = Path(tmp) / "shared_run_map.json"
        map_path.write_text(json.dumps({"source_experiment_id": exp.experiment_id,
                                        "shared_experiment_id": exp_id, "runs": run_map},
                                       sort_keys=True, indent=2))
        files[COMPLETED_RUN]["shared_run_map.json"] = map_path
        verified_now += _sync_run(source, target, root, parent, files[COMPLETED_RUN])
        completed.add(COMPLETED_RUN)
        remote = _check_batch(target, exp_id, parent_id, run_map, completed, readiness)
        if (set(remote) != approved or any(r.info.run_id != run_map[k] or r.info.status != "FINISHED"
                or not r.data.tags.get(VERIFIED) or r.data.tags.get(PARENT) != (None if k == COMPLETED_RUN else parent_id)
                for k, r in remote.items())):
            raise RuntimeError("Shared runs, states or parent links do not match the approved tree")
        evidence = json.loads(files[COMPLETED_RUN]["evidence.json"].read_text())
        destination["final_runs"] = {r["family"]: run_map[r["run_id"]] for r in evidence["final"]}
        _write_checkpoint(checkpoint, destination)
        final = _verify_final(target, evidence, run_map)
        if readiness:
            verify_readiness(target, readiness, check_artifact=True)
        target.set_tag(parent_id, "pbi06.tree_verified", "true")
        split = [details for run in remote.values()
                 for details in json.loads(run.data.tags.get("pbi06.split_files", "{}")).values()]
        logical_count = sum(map(len, files.values()))
        logical_bytes = sum(p.stat().st_size for f in files.values() for p in f.values())
        receipt = {**shared_status(uri), "created": created, "verified_now": verified_now,
                   "runs": len(run_map), "final_runs": final,
                   "artifacts": logical_count, "artifact_bytes": logical_bytes,
                   "split_artifacts": len(split),
                   "physical_files": logical_count + sum(d["parts"] for d in split),
                   "physical_bytes": logical_bytes + sum(d["manifest_bytes"] for d in split),
                   "excluded": excluded, "forest_artifacts_sanitized": sum(
                       run.data.tags.get("family") == "random_forest" for run in children)}
        (workdir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--readiness-file", type=Path, help="JSON containing the verified probe run_id and sha256.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--rename-original", action="store_true",
                       help="Rename the recorded shared experiment in place if it is available.")
    modes.add_argument("--recover-replacement", action="store_true",
                        help="Confirm the configured replacement backend and allow a new publication there.")
    args = parser.parse_args()
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    result = (rename_recorded_shared(uri) if args.rename_original
              else publish(uri, args.workers, recover_replacement=args.recover_replacement,
                           batch_size=args.batch_size,
                           readiness=json.loads(args.readiness_file.read_text()) if args.readiness_file else None))
    print(json.dumps(result, indent=2))
