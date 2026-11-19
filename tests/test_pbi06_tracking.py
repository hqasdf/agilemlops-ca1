import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mlflow.entities import Metric
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from src import pbi06_publish as publisher
from src import pbi06_tracking as tracking
from src import train_hq
from src import pbi06_research as research

REMOTE = "https://shared.example.invalid"


class TrackingTests(unittest.TestCase):
    def test_configured_remote_is_used_and_failure_is_not_retried_locally(self):
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": REMOTE}), \
             patch.object(train_hq.mlflow, "MlflowClient") as client, \
             patch.object(train_hq.mlflow, "set_tracking_uri") as setter, \
             patch.object(train_hq.mlflow, "set_experiment"):
            client.return_value.get_experiment_by_name.side_effect = (
                lambda name: object() if name == tracking.EXPERIMENT else None
            )
            self.assertEqual(train_hq._tracking(publisher.EXPERIMENT, None), REMOTE)
            client.assert_called_once_with(tracking_uri=REMOTE)
            setter.assert_called_once_with(REMOTE)
            client.reset_mock()
            setter.reset_mock()
            client.return_value.get_experiment_by_name.side_effect = ConnectionError("remote unavailable")
            with self.assertRaisesRegex(ConnectionError, "remote unavailable"):
                train_hq._tracking(publisher.EXPERIMENT, None)
            client.assert_called_once_with(tracking_uri=REMOTE)
            setter.assert_not_called()

    def test_canonical_research_passes_remote_destination_before_any_fit(self):
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": REMOTE}), \
             patch.object(research, "_guard", return_value=(None, None)), \
             patch.object(research, "_fingerprint", return_value=("test", {})), \
             patch.object(train_hq, "_tracking", side_effect=ConnectionError("AWS unavailable")) as track, \
             patch.object(research, "fit_preprocessor") as fit:
            experiment = research.research_stages(None, None, None, None, None, None, {}, mode="full")
            with self.assertRaisesRegex(ConnectionError, "AWS unavailable"):
                next(experiment)
            track.assert_called_once_with(publisher.EXPERIMENT, REMOTE)
            fit.assert_not_called()

    def test_readers_use_explicit_clients_without_changing_destination(self):
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": REMOTE}), \
             patch.object(tracking.mlflow, "MlflowClient") as client, \
             patch.object(tracking.mlflow, "set_tracking_uri") as setter, \
             patch.object(tracking.mlflow, "set_experiment") as experiment:
            client.return_value.get_run.side_effect = LookupError("historical test")
            with self.assertRaisesRegex(RuntimeError, "historical"):
                train_hq.load_finalized_results({})
            client.assert_called_once_with(tracking_uri=tracking.LOCAL_URI)
            self.assertEqual(tracking.resolve_tracking_uri(), REMOTE)
            setter.assert_not_called()
            experiment.assert_not_called()

    def test_rename_in_place_requires_recorded_experiment_and_parent(self):
        old = SimpleNamespace(experiment_id="5", lifecycle_stage="active", name=tracking.LEGACY_EXPERIMENT)
        new = SimpleNamespace(experiment_id="5", lifecycle_stage="active", name=tracking.EXPERIMENT)
        client = Mock()
        client.get_experiment_by_name.side_effect = lambda name: old if name == tracking.LEGACY_EXPERIMENT else None
        client.get_experiment.return_value = new
        client.get_run.return_value = SimpleNamespace(
            info=SimpleNamespace(experiment_id="5"),
            data=SimpleNamespace(tags={"fingerprint": tracking.COMPLETED_FINGERPRINT}),
        )
        self.assertIs(tracking.rename_rebuild_experiment(client, "5", tracking.COMPLETED_RUN), new)
        client.rename_experiment.assert_called_once_with("5", "PBI06_HanQian")
        client.rename_experiment.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "Experiment identity"):
            tracking.rename_rebuild_experiment(client, "2", tracking.COMPLETED_RUN)
        client.rename_experiment.assert_not_called()

    def test_source_patch_allowlist_pins_both_reviewed_files(self):
        source = Path(tracking.__file__).parent
        self.assertEqual(set(tracking.OLD_SOURCES), set(tracking.TRACKING_PATCH_SOURCES))
        self.assertEqual({name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in tracking.OLD_SOURCES}, tracking.TRACKING_PATCH_SOURCES)


class PublishingTests(unittest.TestCase):
    def test_resume_rejects_empty_or_recreated_shared_store(self):
        previous = {"tracking_uri": REMOTE, "experiment_id": "2", "parent_run_id": "original"}
        with self.assertRaisesRegex(RuntimeError, "experiment is missing"):
            publisher._check_destination(REMOTE, None, previous)
        experiment = SimpleNamespace(experiment_id="2")
        with self.assertRaisesRegex(RuntimeError, "parent is missing"):
            publisher._check_destination(REMOTE, experiment, previous, {})
        parent = SimpleNamespace(info=SimpleNamespace(run_id="original"))
        publisher._check_destination(REMOTE, experiment, previous, {tracking.COMPLETED_RUN: parent})

    def test_recovery_requires_explicit_confirmation_and_missing_parent(self):
        client = Mock()
        client.get_experiment_by_name.return_value = None
        previous = {"tracking_uri": REMOTE, "experiment_id": "2", "parent_run_id": "original"}
        with self.assertRaisesRegex(RuntimeError, "experiment is missing"):
            publisher._publication_target(client, REMOTE, previous, False)
        with self.assertRaisesRegex(RuntimeError, "still exists"):
            publisher._publication_target(client, REMOTE, previous, True)
        client.get_run.side_effect = MlflowException("gone", error_code=RESOURCE_DOES_NOT_EXIST)
        self.assertEqual(publisher._publication_target(client, REMOTE, previous, True), (None, {}, True))
        client.create_experiment.assert_not_called()

    def test_recovery_refuses_legacy_and_unrelated_experiments(self):
        client = Mock()
        client.get_experiment_by_name.side_effect = (
            lambda name: object() if name == tracking.LEGACY_EXPERIMENT else None
        )
        with self.assertRaisesRegex(RuntimeError, "Rename it in place"):
            publisher._publication_target(client, REMOTE, None, True)
        unrelated = SimpleNamespace(experiment_id="3", tags={})
        client.get_experiment_by_name.side_effect = (
            lambda name: unrelated if name == tracking.EXPERIMENT else None
        )
        with patch.object(publisher, "_existing_copies", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "different experiment"):
                publisher._publication_target(client, REMOTE, None, True)
        client.create_experiment.assert_not_called()

    def test_recovery_can_resume_tagged_partial_tree_and_normal_resume_after_checkpoint(self):
        client = Mock()
        exp = SimpleNamespace(experiment_id="3", tags={publisher.RECOVERY_TAG: tracking.COMPLETED_RUN})
        client.get_experiment_by_name.side_effect = lambda name: exp if name == tracking.EXPERIMENT else None
        client.get_run.side_effect = MlflowException("gone", error_code=RESOURCE_DOES_NOT_EXIST)
        shared = SimpleNamespace(info=SimpleNamespace(run_id="shared-parent"))
        copies = {tracking.COMPLETED_RUN: shared}
        previous = {"tracking_uri": REMOTE, "experiment_id": "2", "parent_run_id": "original"}
        with patch.object(publisher, "_existing_copies", return_value=copies), \
             patch.object(publisher, "_list_runs", return_value=[shared]):
            self.assertEqual(publisher._publication_target(client, REMOTE, previous, True), (exp, copies, True))
            saved = {**previous, "experiment_id": "3", "parent_run_id": "shared-parent"}
            self.assertEqual(publisher._publication_target(client, REMOTE, saved, False), (exp, copies, False))

    def test_batch_check_stops_if_verified_history_disappears(self):
        client = Mock()
        client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="2", lifecycle_stage="active")
        parent = SimpleNamespace(
            info=SimpleNamespace(run_id="shared-parent", status="RUNNING"),
            data=SimpleNamespace(tags={}),
        )
        child = SimpleNamespace(
            info=SimpleNamespace(run_id="shared-child", status="FINISHED"),
            data=SimpleNamespace(tags={publisher.PARENT: "shared-parent", publisher.VERIFIED: "hash"}),
        )
        mapping = {tracking.COMPLETED_RUN: "shared-parent", "local-child": "shared-child"}
        with patch.object(publisher, "_existing_copies", return_value={
            tracking.COMPLETED_RUN: parent, "local-child": child
        }), patch.object(publisher, "verify_readiness") as probe:
            publisher._check_batch(client, "2", "shared-parent", mapping, {"local-child"}, {"run_id": "probe"})
            probe.assert_called_once()
            child.data.tags.pop(publisher.VERIFIED)
            with self.assertRaisesRegex(RuntimeError, "lost its verification state"):
                publisher._check_batch(client, "2", "shared-parent", mapping, {"local-child"}, None)

    def test_replacement_requires_probe_before_reading_local_tree_or_writing(self):
        with patch.object(publisher, "_shared_client") as target, patch.object(publisher, "_source_tree") as source:
            with self.assertRaisesRegex(ValueError, "readiness-file"):
                publisher.publish(REMOTE, recover_replacement=True)
            source.assert_not_called()
            target.return_value.create_experiment.assert_not_called()

    def test_source_reference_preserves_parent_and_prevents_duplicate_creation(self):
        local = SimpleNamespace(
            info=SimpleNamespace(run_id="local-child", start_time=10, run_name="trial"),
            data=SimpleNamespace(tags={publisher.PARENT: "local-parent", "stage": "advanced"}),
        )
        client = Mock()
        new, created = publisher._create_copy(client, "2", {}, local, "shared-parent")
        self.assertTrue(created)
        tags = client.create_run.call_args.kwargs["tags"]
        self.assertEqual(tags[publisher.SOURCE], "local-child")
        self.assertEqual(tags[publisher.PARENT], "shared-parent")
        new.info.start_time, new.info.run_name, new.data.tags = 10, "trial", tags
        client.reset_mock()
        existing, created = publisher._create_copy(client, "2", {"local-child": new}, local, "shared-parent")
        self.assertIs(existing, new)
        self.assertFalse(created)
        client.create_run.assert_not_called()

    def test_metric_events_preserve_steps_and_timestamps(self):
        events = [Metric("f1", .56, 10, 1, run_id="local"), Metric("f1", .57, 11, 2)]
        copied = [Metric("f1", .56, 10, 1, run_id="shared"), Metric("f1", .57, 11, 2)]
        self.assertEqual(publisher._hist_keys(events), publisher._hist_keys(copied))
        self.assertNotEqual(publisher._hist_keys(events), publisher._hist_keys(copied[:1]))

    def test_proxy_size_limit_splits_hashes_and_reassembles(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.joblib"
            source.write_bytes(b"0123456789ABCDEF")
            uploaded = Path(tmp) / "remote"
            output = Path(tmp) / "output"
            output.mkdir()

            def upload(_, filename, artifact_path=None):
                src = Path(filename)
                if src.name == "model.joblib":
                    raise RuntimeError("413 Client Error")
                dst = uploaded / (artifact_path or "") / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)

            def list_files(_, path=None):
                folder = uploaded / (path or "")
                if not folder.is_dir():
                    return []
                return [SimpleNamespace(path=str(p.relative_to(uploaded)), is_dir=p.is_dir())
                        for p in folder.iterdir()]

            def download(_, path, dst_path):
                dest = Path(dst_path) / Path(path).name
                shutil.copyfile(uploaded / path, dest)
                return str(dest)

            client = Mock(log_artifact=Mock(side_effect=upload),
                          list_artifacts=Mock(side_effect=list_files),
                          download_artifacts=Mock(side_effect=download))
            with patch.object(publisher, "CHUNK_BYTES", 4):
                split = publisher._add_files(client, "shared", {"model/model.joblib": source}, str(output))
            self.assertEqual(split["model/model.joblib"]["parts"], 4)
            artifact = publisher.download_published_artifact(client, "shared", "model/model.joblib", output)
            self.assertEqual(artifact.read_bytes(), source.read_bytes())

    def test_explicit_privacy_filter_and_run_scope(self):
        self.assertNotIn("predictions.npz", publisher.APPROVED_FILES)
        self.assertNotIn("orders_base.csv", publisher.APPROVED_FILES)
        self.assertEqual(sum(publisher.EXPECTED_STAGES.values()) + 1, publisher.EXPECTED_RUNS)
        with self.assertRaisesRegex(ValueError, "Potential individual"):
            publisher._safe_text(json.dumps({"customer_id": "example"}), "fixture")


if __name__ == "__main__":
    unittest.main()
