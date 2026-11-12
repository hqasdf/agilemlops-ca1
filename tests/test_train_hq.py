import tempfile
import unittest
import joblib
import json
from pathlib import Path
import mlflow
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import feature_engineering
from src.data_cleaning import clean_orders
from src.feature_engineering import fit_preprocessor, prepare_features
from src.train_hq import model_candidates, rank_results, train_benchmark, validation_metrics, chronological_folds, tune_top_models, advanced_candidates, advanced_tune_top_models, select_f1_threshold, optimize_advanced_thresholds, train_selected_models, load_finalized_results, BASE_DATA_HASHES
from test_data_cleaning import customer, order, label


class TrainingTests(unittest.TestCase):
    def test_metrics_and_ranking(self):
        y = [0, 0, 1, 1]
        scores = [0.1, 0.6, 0.4, 0.9]
        m = validation_metrics(y, scores, 0.5)
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"]), (1, 1, 1, 1))
        self.assertEqual(m["f1"], 0.5)
        self.assertEqual((m["precision"], m["recall"], m["accuracy"]), (0.5, 0.5, 0.5))
        self.assertAlmostEqual(m["average_precision"], 5 / 6)
        self.assertEqual(m["roc_auc"], 0.75)
        self.assertEqual(m["false_positive_rate"], 0.5)
        ranked = rank_results([{"model": "a", "version": "v", "f1": 0.4, "average_precision": 0.9},
                               {"model": "b", "version": "v", "f1": 0.5, "average_precision": 0.1}])
        self.assertEqual(ranked[0]["model"], "b")

    def test_all_active_models_use_default_threshold(self):
        customers = pd.DataFrame([customer()])
        orders = []
        labels = []
        for i in range(40):
            day = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i)
            orders.append(order(order_id=str(i), order_date=day.strftime("%Y-%m-%d"), order_value=i + 1))
            labels.append(label(order_id=str(i), is_fraud=i % 2))
        with patch("src.train_hq._tracking", return_value="local-test"), \
             patch("src.train_hq._log_run", return_value="mock-run"):
            results, _, _ = train_benchmark(customers, pd.DataFrame(orders), pd.DataFrame(labels))
        self.assertEqual(len(results), 6)
        self.assertEqual({r["model"] for r in results}, set(model_candidates()))
        self.assertTrue(all(r["version"] == "baseline" and r["threshold"] == 0.5 for r in results))
        self.assertEqual([r["f1"] for r in results], sorted([r["f1"] for r in results], reverse=True))

    def test_new_models_serialize_with_balanced_sample_weights(self):
        raw = pd.DataFrame([order(order_id=str(i), order_value=i + 1) for i in range(30)])
        features = prepare_features(clean_orders(raw)[0])
        preprocessor = fit_preprocessor(features)
        matrix = preprocessor.transform(features).astype(np.float32)
        labels = np.array([0, 1] * 15)
        weights = np.ones(len(labels))
        for name in ("xgboost", "lightgbm", "catboost"):
            model = model_candidates()[name]
            data = matrix.toarray() if name == "catboost" and hasattr(matrix, "toarray") else matrix
            model.fit(data, labels, sample_weight=weights)
            with tempfile.TemporaryDirectory() as folder:
                path = folder + "/model.joblib"
                joblib.dump(model, path)
                restored = joblib.load(path)
                np.testing.assert_allclose(model.predict_proba(data), restored.predict_proba(data))

    def test_time_aware_tuning_uses_only_inner_folds(self):
        from src.data_cleaning import clean_customers, clean_labels
        from src.feature_engineering import prepare_training_data
        customers = pd.DataFrame([customer()])
        orders = []
        labels = []
        for i in range(60):
            date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i // 2)
            orders.append(order(order_id=str(i), order_date=date.strftime("%Y-%m-%d"), order_value=i + 1))
            labels.append(label(order_id=str(i), is_fraud=i % 2))
        customers = clean_customers(customers)[0]
        orders = clean_orders(pd.DataFrame(orders))[0]
        labels = clean_labels(pd.DataFrame(labels))[0]
        x_train, y_train, x_val, y_val, _, _, report = prepare_training_data(
            customers, orders, labels, include_test=False, include_training_dates=True)
        dates = report.pop("training_dates")
        self.assertTrue(dates.index.equals(x_train.index))
        self.assertNotIn("order_date", x_train.columns)
        folds = chronological_folds(x_train, dates)
        for train_mask, valid_mask in folds:
            self.assertLess(dates[train_mask].max(), dates[valid_mask].min())
            self.assertTrue(set(dates[train_mask]).isdisjoint(set(dates[valid_mask])))
        options = {name: [({}, 1.0)] for name in ("catboost", "lightgbm", "random_forest")}
        from src import train_hq
        original_fit = train_hq.fit_preprocessor
        fitted_indices = []
        weighted_indices = []
        original_weights = train_hq.training_sample_weights
        def check_weights(target):
            self.assertTrue(target.index.intersection(x_val.index).empty)
            weighted_indices.append(target.index)
            return original_weights(target)
        def check_preprocessor(frame):
            self.assertTrue(frame.index.intersection(x_val.index).empty)
            fitted_indices.append(frame.index)
            return original_fit(frame)
        with patch("src.train_hq.tuning_candidates", return_value=options), \
             patch("src.train_hq._tracking", return_value="local-test"), \
             patch("src.train_hq._log_inner_trial", return_value="trial"), \
             patch("src.train_hq._log_run", return_value="final"), \
             patch("src.train_hq.fit_preprocessor", side_effect=check_preprocessor), \
             patch("src.train_hq.training_sample_weights", side_effect=check_weights):
            results, trials, _ = tune_top_models(x_train, y_train, dates, x_val, y_val, report)
        self.assertEqual(len(results), 3)
        self.assertEqual(len(trials), 6)
        self.assertEqual(len(fitted_indices), 3)  # two folds plus full training
        self.assertGreater(len(weighted_indices), 0)
        self.assertTrue(all(index.isin(x_train.index).all() for index in weighted_indices))
        self.assertTrue(all(row["threshold"] == 0.5 for row in results))
        with self.assertRaisesRegex(ValueError, "same unique index"):
            chronological_folds(x_train, dates.iloc[::-1])

    def test_advanced_search_is_reproducible_and_selection_ignores_outer_labels(self):
        from src.data_cleaning import clean_customers, clean_labels
        from src.feature_engineering import prepare_training_data
        self.assertEqual(advanced_candidates(), advanced_candidates())
        self.assertEqual({name: len(options) for name, options in advanced_candidates().items()},
                         {"catboost": 20, "lightgbm": 20, "random_forest": 20})
        for params, strength in advanced_candidates()["random_forest"]:
            self.assertEqual(params.get("class_weight", "balanced"), "balanced" if strength == 1 else None)
        customers = clean_customers(pd.DataFrame([customer()]))[0]
        orders = []
        labels = []
        for i in range(60):
            date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i // 2)
            orders.append(order(order_id=str(i), order_date=date.strftime("%Y-%m-%d"), order_value=i + 1))
            labels.append(label(order_id=str(i), is_fraud=i % 2))
        orders = clean_orders(pd.DataFrame(orders))[0]
        labels = clean_labels(pd.DataFrame(labels))[0]
        x_train, y_train, x_val, y_val, _, _, report = prepare_training_data(
            customers, orders, labels, include_test=False, include_training_dates=True)
        dates = report.pop("training_dates")
        small = {name: [({}, 1.0), (params, 0.5)] for name, params in {
            "catboost": {"iterations": 5}, "lightgbm": {"n_estimators": 5},
            "random_forest": {"n_estimators": 5, "class_weight": None},
        }.items()}
        with patch("src.train_hq.advanced_candidates", return_value=small), \
             patch("src.train_hq._tracking", return_value="local-test"), \
             patch("src.train_hq._log_inner_trial", return_value="trial"), \
             patch("src.train_hq._log_run", return_value="final"):
            first, trials, _ = advanced_tune_top_models(x_train, y_train, dates, x_val, y_val, report)
            second, _, _ = advanced_tune_top_models(x_train, y_train, dates, x_val, 1 - y_val, report)
        chosen1 = {r["model"]: (r["selected_changes"], r["weight_strength"]) for r in first}
        chosen2 = {r["model"]: (r["selected_changes"], r["weight_strength"]) for r in second}
        self.assertEqual(chosen1, chosen2)
        self.assertEqual(len(trials), 12)
        self.assertTrue(all(r["version"] == "advanced_tuned" and r["threshold"] == 0.5 for r in first))

    def test_threshold_selection_and_outer_validation_isolation(self):
        from src.data_cleaning import clean_customers, clean_labels
        from src.feature_engineering import prepare_training_data
        threshold, f1 = select_f1_threshold([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
        self.assertAlmostEqual(threshold, 0.4)
        self.assertAlmostEqual(f1, 0.8)
        customers = clean_customers(pd.DataFrame([customer()]))[0]
        orders, labels = [], []
        for i in range(60):
            date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i // 2)
            orders.append(order(order_id=str(i), order_date=date.strftime("%Y-%m-%d"), order_value=i + 1))
            labels.append(label(order_id=str(i), is_fraud=i % 2))
        orders = clean_orders(pd.DataFrame(orders))[0]
        labels = clean_labels(pd.DataFrame(labels))[0]
        x_train, y_train, x_val, y_val, x_test, y_test, report = prepare_training_data(
            customers, orders, labels, include_test=False, include_training_dates=True)
        dates = report.pop("training_dates")
        self.assertIsNone(x_test)
        self.assertIsNone(y_test)
        preprocessor = fit_preprocessor(x_train)
        x = preprocessor.transform(x_train).astype(np.float32)
        results = []
        with tempfile.TemporaryDirectory() as folder:
            for name in ("lightgbm", "catboost", "random_forest"):
                model = model_candidates()[name]
                data = x.toarray() if name == "catboost" and hasattr(x, "toarray") else x
                model.fit(data, y_train)
                joblib.dump(model, folder + "/" + name + "_model.joblib")
                joblib.dump(preprocessor, folder + "/" + name + "_preprocessor.joblib")
                results.append({"model": name, "run_id": name, "selected_changes": {},
                                "weight_strength": 1.0, "training_seconds": 1.0,
                                "parameters": model.get_params()})
            def artifact(run_id, artifact_path):
                part = "preprocessor" if "preprocessor" in artifact_path else "model"
                return folder + "/" + run_id + "_" + part + ".joblib"
            from src import train_hq
            fitted_indices = []
            original_fit = train_hq.fit_preprocessor
            def check_fold_fit(frame):
                self.assertTrue(frame.index.intersection(x_val.index).empty)
                fitted_indices.append(frame.index)
                return original_fit(frame)
            with patch("src.train_hq._tracking", return_value="local-test"), \
                 patch("src.train_hq.fit_preprocessor", side_effect=check_fold_fit), \
                 patch("src.train_hq.mlflow.artifacts.download_artifacts", side_effect=artifact), \
                 patch("src.train_hq._log_run", return_value="new-run"), \
                 patch("src.train_hq.mlflow.MlflowClient"):
                first, _ = optimize_advanced_thresholds(x_train, y_train, dates, x_val, y_val, report, results)
                second, _ = optimize_advanced_thresholds(x_train, y_train, dates, x_val, 1-y_val, report, results)
        self.assertEqual(len(fitted_indices), 12)  # 3 models × 2 folds × 2 runs
        self.assertTrue(all(index.isin(x_train.index).all() for index in fitted_indices))
        self.assertEqual({r["model"]: r["threshold"] for r in first},
                         {r["model"]: r["threshold"] for r in second})
        self.assertTrue(all(r["version"] == "threshold_optimized" for r in first))
        self.assertTrue(all(0 <= r["threshold"] <= 1 for r in first))

    def test_selected_retraining_logs_thresholds_without_search_or_test(self):
        from src.data_cleaning import clean_customers, clean_labels
        from src.feature_engineering import prepare_training_data
        customers = clean_customers(pd.DataFrame([customer()]))[0]
        order_rows, label_rows = [], []
        for i in range(80):
            date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i // 2)
            order_rows.append(order(order_id=str(i), order_date=date.strftime("%Y-%m-%d"), order_value=i + 1))
            label_rows.append(label(order_id=str(i), is_fraud=i % 2))
        orders = clean_orders(pd.DataFrame(order_rows))[0]
        labels = clean_labels(pd.DataFrame(label_rows))[0]
        x_train, y_train, x_val, y_val, x_test, y_test, report = prepare_training_data(
            customers, orders, labels, include_test=False)
        self.assertIsNone(x_test)
        self.assertIsNone(y_test)
        from src import train_hq
        original_fit = train_hq.fit_preprocessor
        def training_only(frame):
            self.assertTrue(frame.index.equals(x_train.index))
            return original_fit(frame)
        previous_uri = mlflow.get_tracking_uri()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            uri = "sqlite:///" + str(root / "tracking.db")
            mlflow.set_tracking_uri(uri)
            mlflow.create_experiment("test_selected", artifact_location=(root / "artifacts").as_uri())
            try:
                with patch("src.train_hq.fit_preprocessor", side_effect=training_only), \
                     patch("src.train_hq.tune_top_models", side_effect=AssertionError("unexpected search")), \
                     patch("src.train_hq.optimize_advanced_thresholds", side_effect=AssertionError("unexpected search")):
                    results = train_selected_models(x_train, y_train, x_val, y_val, report,
                                                   tracking_uri=uri, experiment_name="test_selected", batch_name="test-batch")
                self.assertEqual({r["model"] for r in results}, {"catboost", "lightgbm"})
                expected = {"catboost": 0.45684071728727366, "lightgbm": 0.5}
                for result in results:
                    self.assertEqual(result["threshold"], expected[result["model"]])
                    run = mlflow.MlflowClient().get_run(result["run_id"])
                    self.assertIn("test-batch", run.data.tags["mlflow.runName"])
                    model_file = mlflow.artifacts.download_artifacts(run_id=result["run_id"], artifact_path="model/model.joblib")
                    prep_file = mlflow.artifacts.download_artifacts(run_id=result["run_id"], artifact_path="model/preprocessor.joblib")
                    info_file = mlflow.artifacts.download_artifacts(run_id=result["run_id"], artifact_path="model/inference.json")
                    self.assertEqual(json.loads(Path(info_file).read_text())["threshold"], expected[result["model"]])
                    model = joblib.load(model_file)
                    preprocessor = joblib.load(prep_file)
                    x = preprocessor.transform(x_val).astype(np.float32)
                    if result["model"] == "catboost" and hasattr(x, "toarray"):
                        x = x.toarray()
                    reproduced = validation_metrics(y_val, model.predict_proba(x)[:, 1], expected[result["model"]])
                    self.assertAlmostEqual(reproduced["f1"], result["metrics"]["f1"])
                    self.assertTrue(result["inference_uri"].endswith("/model/inference.json"))
            finally:
                mlflow.set_tracking_uri(previous_uri)

    def test_missing_historical_run_is_a_clear_error(self):
        report = {"train_first_date": "2022-11-12", "train_last_date": "2023-03-14",
                  "validation_first_date": "2023-03-15", "validation_last_date": "2023-04-13",
                  "training_rows": 660773, "validation_rows": 144978}
        with patch("src.train_hq._tracking"), patch("src.train_hq.data_hashes", return_value=BASE_DATA_HASHES), \
             patch("src.train_hq.mlflow.MlflowClient") as client:
            client.return_value.get_run.side_effect = FileNotFoundError("missing")
            with self.assertRaisesRegex(RuntimeError, "historical MLflow runs or artifacts are missing"):
                load_finalized_results(report)

    def test_saved_model_and_preprocessor_reload(self):
        raw = pd.DataFrame([order(order_id=str(i), order_value=i + 1) for i in range(6)])
        orders = clean_orders(raw)[0]
        features = prepare_features(orders)
        preprocessor = fit_preprocessor(features)
        matrix = preprocessor.transform(features)
        model = LogisticRegression(random_state=42).fit(matrix, [0, 1, 0, 1, 0, 1])
        with tempfile.TemporaryDirectory() as folder:
            joblib.dump(model, folder + "/model.joblib")
            joblib.dump(preprocessor, folder + "/preprocessor.joblib")
            loaded_model = joblib.load(folder + "/model.joblib")
            loaded_preprocessor = joblib.load(folder + "/preprocessor.joblib")
            np.testing.assert_allclose(model.predict_proba(matrix),
                                       loaded_model.predict_proba(loaded_preprocessor.transform(features)))

    def test_no_test_features_and_callable_reproducible(self):
        customers = pd.DataFrame([customer()])
        orders = []
        labels = []
        for i in range(40):
            day = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i)
            orders.append(order(order_id=str(i), order_date=day.strftime("%Y-%m-%d"), order_value=i + 1))
            labels.append(label(order_id=str(i), is_fraud=i % 2))
        orders = pd.DataFrame(orders)
        labels = pd.DataFrame(labels)
        original = feature_engineering.prepare_features
        def no_test_features(frame):
            self.assertLess(frame["order_date"].max(), pd.Timestamp("2023-02-04"))
            return original(frame)
        candidates = {"logistic_regression": LogisticRegression(class_weight="balanced", random_state=42)}
        with patch.object(feature_engineering, "prepare_features", side_effect=no_test_features), \
             patch("src.train_hq.model_candidates", return_value=candidates), \
             patch("src.train_hq._tracking", return_value="local-test"), \
             patch("src.train_hq._log_run", return_value="mock-run"):
            first, report, _ = train_benchmark(customers, orders, labels)
            second, report2, _ = train_benchmark(customers, orders, labels)
        self.assertEqual(report, report2)
        self.assertLess(report["train_last_date"], report["validation_first_date"])
        self.assertLess(report["validation_last_date"], report["test_first_date"])
        for a, b in zip(first, second):
            for key in ["model", "version", "f1", "precision", "recall", "average_precision", "threshold"]:
                self.assertEqual(a[key], b[key])
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["version"], "baseline")
        self.assertEqual(first[0]["threshold"], 0.5)
        self.assertEqual(set(model_candidates()), {"logistic_regression", "random_forest", "hist_gradient_boosting", "xgboost", "lightgbm", "catboost"})


if __name__ == "__main__":
    unittest.main()
