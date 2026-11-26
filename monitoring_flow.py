"""Monitor a simulation batch with Evidently and store scalar results in PostgreSQL."""
import argparse
from pathlib import Path
import sys
from uuid import UUID, uuid4

from prefect import flow, task

from config import OUT
from tasks.monitoring import (
    build_metric_rows, calculate_drift, calculate_performance, check_parameters,
    load_batch_manifest, load_current_batch, load_reference, persist_metrics,
    prepare_summary, save_reports,
)


@task(name="Load Monitoring Inputs")
def load_inputs(manifest_path, reference_path, training_path, model_name, model_version):
    from database.connection import SessionLocal
    manifest = load_batch_manifest(manifest_path)
    reference, info = load_reference(reference_path, training_path)
    with SessionLocal() as session:
        current = load_current_batch(session, manifest, model_name, model_version)
    return manifest, reference, info, current


@task(name="Calculate Evidently Reports")
def calculate_reports(reference, current, mode, parameters):
    drift, drift_report = calculate_drift(reference, current, **{
        k: v for k, v in parameters.items() if k != "min_labeled"
    })
    performance, performance_report = calculate_performance(current, mode, parameters["min_labeled"])
    return drift, performance, drift_report, performance_report


@task(name="Persist Monitoring Run", retries=0)
def store_results(summary, directory, drift_report, performance_report):
    from database.connection import engine
    # A failed DB commit leaves inspectable reports and can be safely retried.
    save_reports(directory, summary, drift_report, performance_report)
    inserted = persist_metrics(engine, build_metric_rows(summary))
    return inserted


@flow(name="AI4I Monitoring Flow", log_prints=True)
def monitoring_flow(batch_id, reference_model_name, reference_model_version,
                    manifest_dir=OUT / "traffic_batches", reference_path=OUT / "reference.csv",
                    training_path=OUT / "train.csv", output_dir=OUT / "monitoring",
                    model_name=None, model_version=None, run_id=None,
                    drift_share=0.5, min_samples=100, min_labeled=30,
                    numerical_threshold=0.1, categorical_threshold=0.1):
    batch_id = str(UUID(batch_id))
    run_id = str(UUID(run_id)) if run_id else str(uuid4())
    parameters = dict(drift_share=drift_share, min_samples=min_samples, min_labeled=min_labeled,
                      numerical_threshold=numerical_threshold, categorical_threshold=categorical_threshold)
    check_parameters(**parameters)
    manifest, reference, info, current = load_inputs(
        Path(manifest_dir) / f"{batch_id}.json", reference_path, training_path, model_name, model_version)
    if manifest["batch_id"] != batch_id:
        raise ValueError("Manifest filename and batch ID disagree")
    identities = current[["model_name", "model_version"]].drop_duplicates()
    if len(identities) and tuple(identities.iloc[0]) != (reference_model_name, reference_model_version):
        raise ValueError("Selected model does not match the operator-confirmed reference model")
    drift, performance, drift_report, performance_report = calculate_reports(reference, current, manifest["mode"], parameters)
    summary = prepare_summary(manifest, current, info, reference_model_name, reference_model_version,
                              run_id, parameters, drift, performance)
    directory = Path(output_dir) / run_id / summary["input_fingerprint"]
    inserted = store_results(summary, directory, drift_report, performance_report)
    print(f"Run ID: {run_id}; batch: {batch_id}; mode: {manifest['mode']}")
    print(f"Predictions: {len(current)}; drift: {drift['status']}; "
          f"drifted features: {drift['number_of_drifted_features']}/{len(reference.columns)}")
    print(f"Performance: {performance['status']}; labels: {performance['labeled_predictions']}")
    print(f"Metric rows inserted: {inserted}; reports: {directory}")
    if summary["alert"]:
        print("DRIFT ALERT: dataset drift threshold reached. Review the batch; retraining is not automatic.")
    return {**summary, "inserted_metrics": inserted, "report_directory": str(directory)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--reference-model-name", required=True,
                        help="Confirm which deployed model the saved reference belongs to")
    parser.add_argument("--reference-model-version", required=True)
    parser.add_argument("--model-name", help="Optional filter for a batch containing multiple models")
    parser.add_argument("--model-version", help="Optional filter for a batch containing multiple versions")
    parser.add_argument("--run-id", help="Reuse the same UUID for an identical retry")
    parser.add_argument("--manifest-dir", type=Path, default=OUT / "traffic_batches")
    parser.add_argument("--reference-path", type=Path, default=OUT / "reference.csv")
    parser.add_argument("--training-path", type=Path, default=OUT / "train.csv")
    parser.add_argument("--output-dir", type=Path, default=OUT / "monitoring")
    parser.add_argument("--drift-share", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--min-labeled", type=int, default=30)
    parser.add_argument("--numerical-threshold", type=float, default=0.1)
    parser.add_argument("--categorical-threshold", type=float, default=0.1)
    args = parser.parse_args(argv)
    try:
        UUID(args.batch_id)
        if args.run_id:
            UUID(args.run_id)
        check_parameters(args.drift_share, args.min_samples, args.min_labeled,
                         args.numerical_threshold, args.categorical_threshold)
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        monitoring_flow(**vars(args))
    except Exception:
        print("Monitoring failed; inspect the Prefect error and report directory. "
              "Check inputs, database connectivity, and migration status.", file=sys.stderr)
        sys.exit(1)
