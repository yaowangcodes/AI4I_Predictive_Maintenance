"""Generate one batch, run Evidently, and print its Grafana dashboard link."""
import argparse
import json
from pathlib import Path
import sys
from urllib.parse import urlencode

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUT
from monitoring_flow import monitoring_flow
from scripts.simulate_traffic import parse_args as traffic_args, run_simulation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["normal", "drift"], required=True)
    parser.add_argument("--reference-model-name", required=True)
    parser.add_argument("--reference-model-version", required=True)
    parser.add_argument("--n", default="500")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--delay", default="0.02")
    parser.add_argument("--drift-strength", default="1")
    parser.add_argument("--api-url", default=None)
    args = parser.parse_args(argv)
    cli = ["--mode", args.mode, "--n", args.n, "--seed", args.seed, "--delay", args.delay,
           "--drift-strength", args.drift_strength]
    if args.api_url:
        cli += ["--api-url", args.api_url]
    simulation = traffic_args(cli)
    with requests.Session() as session:
        batch = run_simulation(simulation, session)
    if batch["summary"]["failed_predictions"] or batch["summary"]["failed_feedback"]:
        raise RuntimeError("Simulation had failures; inspect its manifest before monitoring")
    result = monitoring_flow(batch["batch_id"], args.reference_model_name, args.reference_model_version)
    link = "http://127.0.0.1:3000/d/ai4i-monitoring?" + urlencode({
        "var-model": args.reference_model_name, "var-version": args.reference_model_version,
        "var-mode": args.mode, "var-batch": batch["batch_id"], "from": "now-1h", "to": "now"})
    evidence = {"batch_id": batch["batch_id"], "run_id": result["run_id"],
                "mode": args.mode, "dataset_drift": result["drift"]["dataset_drift"],
                "drifted_features": result["drift"]["number_of_drifted_features"],
                "dashboard_url": link, "report_directory": result["report_directory"]}
    (OUT / "monitoring" / f"demo-latest-{args.mode}.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    return evidence


if __name__ == "__main__":
    main()
