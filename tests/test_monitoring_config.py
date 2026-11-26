import json
from pathlib import Path

import yaml

from monitoring.build_dashboard import build

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_matches_generator_and_sources():
    dashboard = json.loads((ROOT / 'monitoring/grafana/dashboards/ai4i-monitoring.json').read_text())
    assert dashboard == build()
    assert len({p['id'] for p in dashboard['panels']}) == len(dashboard['panels'])
    sources = {t['datasource']['uid'] for p in dashboard['panels'] for t in p.get('targets', [])}
    assert sources == {'ai4i-postgres', 'ai4i-prometheus'}
    assert {v['name'] for v in dashboard['templating']['list']} == {'model', 'version', 'mode', 'batch'}
    for panel in dashboard['panels']:
        for target in panel.get('targets', []):
            if 'rawSql' in target:
                assert 'run_id IS NOT NULL' in target['rawSql']
                assert 'model_version = ${version:sqlstring}' in target['rawSql']
                assert 'COALESCE' not in target['rawSql'].upper()
                assert '$__timeFilter(calculated_at)' in target['rawSql']


def test_performance_uses_one_latest_run_and_keeps_nulls():
    for panel in build()['panels']:
        if panel['title'] in ['Accuracy', 'Precision', 'Recall', 'F1 score', 'ROC-AUC']:
            sql = panel['targets'][0]['rawSql']
            assert "metric_name = 'dataset_drift'" in sql
            assert 'run_id=(SELECT run_id FROM latest)' in sql
            assert 'metric_value IS NOT NULL' not in sql
            assert panel['fieldConfig']['defaults']['noValue'] == 'Not available'


def test_alert_uses_stored_decision_latest_model_and_explicit_no_data():
    config = yaml.safe_load((ROOT / 'monitoring/grafana/provisioning/alerting/drift.yml').read_text())
    group = config['groups'][0]
    rule = group['rules'][0]
    assert group['interval'] == '10s'
    assert rule['noDataState'] == 'NoData' and rule['execErrState'] == 'Error'
    sql = rule['data'][0]['model']['rawSql']
    assert 'DISTINCT ON (model_name, model_version)' in sql
    assert "metric_name = 'dataset_drift' AND run_id IS NOT NULL" in sql
    assert "ELSE NULL" in sql and "INTERVAL '24 hours'" in sql
    assert 'simulation_mode' not in sql and 'batch_id' not in sql
    assert '${' not in sql
    panel_id = int(rule['annotations']['__panelId__'])
    assert any(p['id'] == panel_id and p['title'] == 'Dataset drift status' for p in build()['panels'])


def test_scrape_targets_and_compose_storage():
    standard = yaml.safe_load((ROOT / 'monitoring/prometheus.yml').read_text())
    host = yaml.safe_load((ROOT / 'monitoring/prometheus.host.yml').read_text())
    for config in [standard, host]:
        assert config['global']['scrape_interval'] == '5s'
        assert config['scrape_configs'][0]['metrics_path'] == '/metrics'
        assert config['scrape_configs'][0]['job_name'] == 'ai4i-api'
    assert standard['scrape_configs'][0]['static_configs'][0]['targets'] == ['api:8000']
    assert host['scrape_configs'][0]['static_configs'][0]['targets'] == ['host.docker.internal:8000']
    compose = yaml.safe_load((ROOT / 'docker-compose.yml').read_text())
    assert 'postgres_data:/var/lib/postgresql/data' in compose['services']['postgres']['volumes']
    for service in ['api','grafana','prometheus']:
        assert all(port.startswith('127.0.0.1:') for port in compose['services'][service]['ports'])
    assert compose['services']['api']['environment']['POSTGRES_HOST'] == 'postgres'


def test_datasource_passwords_are_environment_references():
    source = yaml.safe_load((ROOT / 'monitoring/grafana/provisioning/datasources/sources.yml').read_text())
    pg = next(s for s in source['datasources'] if s['uid'] == 'ai4i-postgres')
    assert pg['secureJsonData']['password'] == '$GRAFANA_DB_PASSWORD'
    assert pg['user'] == '$GRAFANA_DB_USER'
