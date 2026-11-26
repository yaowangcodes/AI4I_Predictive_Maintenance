"""Send reproducible held-out traffic through the existing prediction API."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
from uuid import uuid4

import pandas as pd
import requests

# Support both direct execution and imports from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FEATURES, OUT, SEED, TARGET
from api.schemas import PredictionRequest, PredictionResponse

FIELD_MAP = {
    "Type": "Type",
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
}
DRIFT = {
    "air_temperature": 5,
    "process_temperature": 5,
    "torque": 15,
    "tool_wear": 50,
}


class DatasetError(ValueError):
    """A dataset problem safe to report without printing source rows."""


def build_payload(row, mode="normal", drift_strength=1.0):
    payload = {field: row[column] for column, field in FIELD_MAP.items()}
    # Validate before converting, so fractional integer features are not truncated.
    payload = PredictionRequest.model_validate(payload).model_dump()
    if mode == "drift":
        for field, shift in DRIFT.items():
            increment = shift * drift_strength
            # The API requires whole minutes; round the increment to nearest minute.
            payload[field] += round(increment) if field == "tool_wear" else increment
    return PredictionRequest.model_validate(payload).model_dump()


def load_dataset(path):
    data = pd.read_csv(path)
    if not set(FEATURES + [TARGET]).issubset(data.columns):
        raise DatasetError("Dataset is missing required feature or target columns")
    if data.empty:
        raise DatasetError("Dataset contains no observations")
    if not data[TARGET].isin([0, 1]).all():
        raise DatasetError("Every feature row must contain a valid binary target; check feature/label alignment")
    for position, (_, row) in enumerate(data.iterrows()):
        try:
            build_payload(row)
        except ValueError as error:
            raise DatasetError(f"Dataset row {position} contains features rejected by the API schema") from error
    return data


def sample_rows(data, n, seed):
    # Sample entire rows with replacement; labels never undergo separate sampling.
    return data.sample(n=n, replace=True, random_state=seed)


def save_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def post_json(session, url, payload, timeout):
    response = session.post(url, json=payload, timeout=timeout, allow_redirects=False)
    if not 200 <= response.status_code < 300:
        raise ValueError(f"HTTP {response.status_code}")
    return response.json()


def error_message(error):
    # Do not print response bodies, URLs, or exception details containing secrets.
    if isinstance(error, requests.Timeout):
        return "Request timed out; server outcome may be unknown"
    if isinstance(error, requests.RequestException):
        return "HTTP connection/request failure; server outcome may be unknown"
    if str(error).startswith("HTTP ") and str(error)[5:].isdigit():
        return str(error)
    return "Unexpected API response"


def run_simulation(args, session):
    data = load_dataset(args.dataset)
    sampled = sample_rows(data, args.n, args.seed)
    # Preflight all selected inputs before generating any remote records.
    payloads = [build_payload(row, args.mode, args.drift_strength)
                for _, row in sampled.iterrows()]
    batch_id = str(uuid4())
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    path = args.manifest_dir / f"{batch_id}.json"
    manifest = {
        "batch_id": batch_id,
        "mode": args.mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "requested_predictions": args.n,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "sampling": "with replacement; source_row is zero-based CSV data row",
        "drift_strength": args.drift_strength,
        "drift_offsets": {key: (round(value * args.drift_strength)
                                if key == "tool_wear" else value * args.drift_strength)
                          for key, value in DRIFT.items()} if args.mode == "drift" else {},
        "status": "running",
        "prediction_ids": [],
        "records": [],
    }
    summary = dict(attempted_predictions=0, successful_predictions=0,
                   failed_predictions=0, successful_feedback=0, failed_feedback=0)
    manifest["summary"] = summary
    save_manifest(path, manifest)
    started = time.perf_counter()
    latencies = []
    try:
        for position, ((source_row, row), payload) in enumerate(zip(sampled.iterrows(), payloads)):
            record = {"source_row": int(source_row), "prediction_status": "pending",
                      "feedback_status": "not_attempted"}
            manifest["records"].append(record)
            summary["attempted_predictions"] += 1
            save_manifest(path, manifest)
            request_start = time.perf_counter()
            try:
                result = post_json(session, args.api_url + "/predict", payload, args.timeout)
                parsed = PredictionResponse.model_validate(result, strict=True)
                if (parsed.prediction_id <= 0 or not math.isfinite(parsed.failure_probability)
                        or not 0 <= parsed.failure_probability <= 1
                        or not parsed.model_name or not parsed.model_version
                        or parsed.prediction_id in manifest["prediction_ids"]):
                    raise ValueError("Invalid prediction response")
            except (requests.RequestException, ValueError, TypeError) as error:
                summary["failed_predictions"] += 1
                record.update(prediction_status="failed_or_unknown", error=error_message(error))
                print(f"Prediction {position + 1}: {record['error']}")
            else:
                summary["successful_predictions"] += 1
                record.update(prediction_status="success", **parsed.model_dump())
                manifest["prediction_ids"].append(parsed.prediction_id)
            finally:
                latency = time.perf_counter() - request_start
                latencies.append(latency)
                record["prediction_latency_seconds"] = latency
                save_manifest(path, manifest)

            if record["prediction_status"] == "success" and args.mode == "normal":
                feedback = {"prediction_id": parsed.prediction_id, "actual_label": int(row[TARGET])}
                record["feedback_status"] = "pending"
                save_manifest(path, manifest)
                try:
                    result = post_json(session, args.api_url + "/feedback", feedback, args.timeout)
                    if (not isinstance(result, dict) or result.get("status") != "success"
                            or result.get("prediction_id") != feedback["prediction_id"]
                            or result.get("actual_label") != feedback["actual_label"]):
                        raise ValueError("Invalid feedback response")
                except (requests.RequestException, ValueError, TypeError) as error:
                    summary["failed_feedback"] += 1
                    record.update(feedback_status="failed_or_unknown", feedback_error=error_message(error))
                    print(f"Feedback {position + 1}: {record['feedback_error']}")
                else:
                    summary["successful_feedback"] += 1
                    record.update(feedback_status="success", actual_label=feedback["actual_label"])
                save_manifest(path, manifest)
            if position + 1 < args.n:
                time.sleep(args.delay)
        manifest["status"] = "completed"
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["average_prediction_latency_seconds"] = sum(latencies) / len(latencies) if latencies else 0
        if manifest["status"] == "running":
            manifest["status"] = "interrupted"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_manifest(path, manifest)
        print(json.dumps({"batch_id": batch_id, "mode": args.mode, **summary,
                          "manifest": str(path)}, indent=2))
    return manifest


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["normal", "drift"], default="normal")
    parser.add_argument("--n", type=positive_int, default=5)
    parser.add_argument("--delay", type=nonnegative_float, default=0.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--drift-strength", type=nonnegative_float, default=1.0)
    parser.add_argument("--api-url", default=os.getenv("API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--timeout", type=nonnegative_float, default=10.0)
    parser.add_argument("--dataset", type=Path, default=OUT / "test.csv")
    parser.add_argument("--manifest-dir", type=Path, default=OUT / "traffic_batches")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.timeout == 0:
        parser.error("--timeout must be positive")
    try:
        url = urlsplit(args.api_url)
        valid = (url.scheme in {"http", "https"} and url.hostname and url.port != 0
                 and not url.username and not url.password and not url.query and not url.fragment)
    except ValueError:
        valid = False
    if not valid:
        parser.error("--api-url must be an HTTP(S) base URL without credentials, query, or fragment")
    args.api_url = args.api_url.rstrip("/")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        with requests.Session() as session:
            manifest = run_simulation(args, session)
    except KeyboardInterrupt:
        return 130
    except DatasetError as error:
        print(str(error), file=sys.stderr)
        return 1
    except FileNotFoundError:
        print("Dataset or manifest directory not found; check --dataset and --manifest-dir.", file=sys.stderr)
        return 1
    except (OSError, ValueError, OverflowError):
        print("Simulation could not start or save its manifest. Check dataset columns, binary labels, "
              "API-valid feature values, drift magnitude, and file permissions.", file=sys.stderr)
        return 1
    return int(bool(manifest["summary"]["failed_predictions"] or manifest["summary"]["failed_feedback"]))


if __name__ == "__main__":
    sys.exit(main())
