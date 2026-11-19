"""Resolve write destinations and isolate historical evidence readers."""
import os
from pathlib import Path

import mlflow

LOCAL_URI = "sqlite:///" + str(
    (Path(__file__).resolve().parents[1] / "artifacts/mlflow/mlflow.db")
)
EXPERIMENT = "PBI06_HanQian"
LEGACY_EXPERIMENT = "PBI06_SevenFamily_Rebuild"
COMPLETED_RUN = "939506bd3eb6434c8489d5871ea9778e"
COMPLETED_FINGERPRINT = "c52849e7a9a22643c6a2ce9a0305ee8aa34f8363ff0a4029e1e8cd39b6cbd1e3"

# Exact pre-tracking sources for the historical run and exact reviewed current sources.
OLD_SOURCES = {
    "pbi06_research.py": "581dec8338d6d8119bfe3b6818aee8f971d5ebfffc9526c478e7245acfb58306",
    "train_hq.py": "12abe4239c00532f23a1a4cc7fa1192c4b83d312a1e0bc8b564235436696ab19",
}
TRACKING_PATCH_SOURCES = {'pbi06_research.py': 'e3fd43ce561974208ea49d58211fec86f91de60aa17a5c612ef793cd3408444f', 'train_hq.py': '1bf58fe0f11b5ca46958976f40ea0330119ea2e7769f13cefcc867c8f0eb5277'}


def resolve_tracking_uri(uri=None):
    return uri or os.environ.get("MLFLOW_TRACKING_URI") or LOCAL_URI


def evidence_client(uri=None):
    return mlflow.MlflowClient(tracking_uri=uri or LOCAL_URI)


def rename_rebuild_experiment(client, expected_id, parent_id, *, shared=False):
    """Rename only the known rebuild experiment, or confirm its existing name."""
    old = client.get_experiment_by_name(LEGACY_EXPERIMENT)
    new = client.get_experiment_by_name(EXPERIMENT)
    if old and new:
        raise RuntimeError("Both PBI 06 experiment names exist. Resolve the collision before renaming.")
    experiment = old or new
    if experiment is None:
        return None
    if experiment.experiment_id != str(expected_id) or experiment.lifecycle_stage != "active":
        raise RuntimeError("Experiment identity does not match the recorded rebuild")
    parent = client.get_run(parent_id)
    expected_tag = ("pbi06.source_run_id", COMPLETED_RUN) if shared else (
        "fingerprint", COMPLETED_FINGERPRINT)
    if (parent.info.experiment_id != experiment.experiment_id
            or parent.data.tags.get(expected_tag[0]) != expected_tag[1]):
        raise RuntimeError("Parent identity does not match the recorded rebuild")
    if old:
        client.rename_experiment(experiment.experiment_id, EXPERIMENT)
        return client.get_experiment(experiment.experiment_id)
    return experiment
