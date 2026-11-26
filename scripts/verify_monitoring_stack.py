"""Verify local Prometheus/Grafana health and wait for a demonstrated alert state."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from dotenv import dotenv_values
import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expect', choices=['normal', 'drift', 'nodata'], required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    settings = dotenv_values(ROOT / '.env.monitoring')
    session = requests.Session()
    session.auth = (settings['GRAFANA_ADMIN_USER'], settings['GRAFANA_ADMIN_PASSWORD'])
    base = 'http://127.0.0.1:3000'
    for uid in ['ai4i-postgres', 'ai4i-prometheus']:
        response = session.get(f'{base}/api/datasources/uid/{uid}/health', timeout=10)
        response.raise_for_status()
        assert response.json()['status'] == 'OK'
    deadline = time.monotonic() + 60
    while True:
        response = session.get(f'{base}/api/prometheus/grafana/api/v1/rules', timeout=10)
        response.raise_for_status()
        rule = next(rule for group in response.json()['data']['groups'] for rule in group['rules']
                    if rule['uid'] == 'ai4i-dataset-drift')
        states = [a['state'] for a in rule.get('alerts', [])]
        correct = ((args.expect == 'drift' and rule['state'] == 'firing' and 'Alerting' in states)
                   or (args.expect == 'normal' and rule['state'] == 'inactive' and rule.get('health') == 'ok'
                       and all(state == 'Normal' for state in states))
                   or (args.expect == 'nodata' and ('NoData' in states or rule.get('health') == 'nodata')))
        if correct:
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f"Expected {args.expect}; got {rule['state']}, {rule.get('health')}, {states}")
        time.sleep(2)
    response = requests.get('http://127.0.0.1:9090/api/v1/query',
                            params={'query': 'up{job="ai4i-api"}'}, timeout=10)
    response.raise_for_status()
    assert all(float(r['value'][1]) == 1 for r in response.json()['data']['result'])
    assert response.json()['data']['result']
    evidence = {'checked_at': datetime.now(timezone.utc).isoformat(), 'expected': args.expect,
                'prometheus_target': 'up', 'datasources': 'healthy',
                'rule_state': rule['state'], 'health': rule.get('health'), 'instances': states,
                'annotations': [a.get('annotations') for a in rule.get('alerts', [])]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
