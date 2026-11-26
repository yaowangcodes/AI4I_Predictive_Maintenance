"""Generate the checked-in Grafana dashboard JSON (no credentials or live writes)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROM = {"type": "prometheus", "uid": "ai4i-prometheus"}
PG = {"type": "grafana-postgresql-datasource", "uid": "ai4i-postgres"}
FILTER = """run_id IS NOT NULL
 AND model_name = ${model:sqlstring} AND model_version = ${version:sqlstring}
 AND (${mode:sqlstring} = '__all' OR simulation_mode = ${mode:sqlstring})
 AND (${batch:sqlstring} = '__all' OR batch_id = ${batch:sqlstring})
 AND $__timeFilter(calculated_at)"""
LATEST = f"""WITH latest AS (SELECT run_id FROM monitoring_metrics WHERE {FILTER}
 AND metric_name = 'dataset_drift' ORDER BY calculated_at DESC, id DESC LIMIT 1)"""


def sql_target(query, format="table"):
    return {"refId": "A", "datasource": PG, "rawSql": query, "format": format, "editorMode": "code"}


def prom_target(expression, legend="", instant=False):
    return {"refId": "A", "datasource": PROM, "expr": expression,
            "legendFormat": legend, "instant": instant, "range": not instant}


def build():
    panels = []

    def panel(title, kind, x, y, w, h, targets=None, unit="short", description=""):
        item = {"id": len(panels)+1, "title": title, "type": kind,
                "gridPos": {"x": x, "y": y, "w": w, "h": h},
                "description": description, "targets": targets or [],
                "fieldConfig": {"defaults": {"unit": unit, "noValue": "Not available",
                    "color": {"mode": "palette-classic"}, "mappings": [{"type": "special", "options": {"match": "null", "result": {"text": "Not available", "color": "gray"}}}], "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}}, "overrides": []},
                "options": {}}
        if kind == "stat":
            item["options"] = {"reduceOptions": {"calcs": ["last"], "fields": "", "values": False},
                "orientation": "auto", "textMode": "auto", "colorMode": "value", "graphMode": "none"}
        if kind == "timeseries":
            item["options"] = {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}}
            item["fieldConfig"]["defaults"]["custom"] = {"lineWidth": 2, "fillOpacity": 10, "spanNulls": False}
        if unit == "percentunit":
            item["fieldConfig"]["defaults"].update(min=0, max=1)
        if kind == "row":
            item.update(collapsed=False, panels=[])
        panels.append(item)
        return item

    note = panel("AI4I · Predictive maintenance", "text", 0, 0, 24, 3)
    note["options"] = {"mode": "markdown", "content": "### Service health → distribution shift → failure detection\nAPI metrics cover all application traffic. Evidently panels show only manifest-tracked batches for the selected model/version. Latest-run cards respect the time, mode and batch filters. **Missing labels remain unavailable.**"}
    panel("01 / API operations", "row", 0, 3, 24, 1)
    stat_specs = [
        ("HTTP requests · total", 'sum(ai4i_http_requests_total{job="ai4i-api"})', "short"),
        ("Prediction requests", 'sum(ai4i_prediction_requests_total{job="ai4i-api"})', "short"),
        ("HTTP errors · 4xx + 5xx", 'sum(ai4i_http_errors_total{job="ai4i-api"}) or vector(0)', "short"),
        ("Average latency · 5m", 'sum(rate(ai4i_http_request_duration_seconds_sum{job="ai4i-api"}[5m])) / sum(rate(ai4i_http_request_duration_seconds_count{job="ai4i-api"}[5m]))', "s"),
        ("P95 latency · 5m", 'histogram_quantile(0.95, sum by (le) (rate(ai4i_http_request_duration_seconds_bucket{job="ai4i-api"}[5m])))', "s"),
        ("API scrape availability", 'min(up{job="ai4i-api"})', "short")]
    for index, (title, expr, unit) in enumerate(stat_specs):
        p = panel(title, "stat", index*4, 4, 4, 4, [prom_target(expr, instant=True)], unit)
        if index == 5:
            p["fieldConfig"]["defaults"].update(mappings=[{"type": "value", "options": {"0": {"text": "DOWN", "color": "red"}, "1": {"text": "UP", "color": "green"}}}])
    panel("Request rate by endpoint", "timeseries", 0, 8, 8, 6,
          [prom_target('sum by (endpoint) (rate(ai4i_http_requests_total{job="ai4i-api"}[1m]))', '{{endpoint}}')], "reqps")
    panel("HTTP error rate · 4xx + 5xx", "timeseries", 8, 8, 8, 6,
          [prom_target('(sum(rate(ai4i_http_errors_total{job="ai4i-api"}[1m])) or vector(0)) / clamp_min(sum(rate(ai4i_http_requests_total{job="ai4i-api"}[1m])), 0.000001)', "HTTP error fraction")], "percentunit")
    targets = [prom_target('sum(rate(ai4i_http_requests_total{job="ai4i-api",status=~"[23].."}[1m])) or vector(0)', 'Successful (2xx/3xx)'),
               {**prom_target('sum(rate(ai4i_http_requests_total{job="ai4i-api",status=~"[45].."}[1m])) or vector(0)', 'Failed (4xx/5xx)'), "refId": "B"}]
    panel("Successful vs failed requests", "timeseries", 16, 8, 8, 6, targets, "reqps")
    panel("02 / Data drift · latest selected monitoring run", "row", 0, 14, 24, 1)
    for index, (name, title, unit) in enumerate([
        ("dataset_drift", "Dataset drift status", "short"),
        ("number_of_features", "Monitored features", "short"),
        ("number_of_drifted_features", "Drifting features", "short"),
        ("share_of_drifted_features", "Share of drifting features", "percentunit")]):
        query = LATEST + f" SELECT metric_value AS \"{title}\" FROM monitoring_metrics WHERE run_id=(SELECT run_id FROM latest) AND metric_name='{name}'"
        p = panel(title, "stat", index*6, 15, 6, 4, [sql_target(query)], unit)
        if name == "dataset_drift":
            p["fieldConfig"]["defaults"]["mappings"] = [{"type": "value", "options": {"0": {"text": "NORMAL", "color": "green"}, "1": {"text": "DRIFT", "color": "red"}}}]
    panel("Drift share vs configured threshold", "timeseries", 0, 19, 12, 8,
          [sql_target(f'SELECT calculated_at AS time, metric_value AS "Drifted share", threshold AS "Configured threshold" FROM monitoring_metrics WHERE {FILTER} AND metric_name=\'share_of_drifted_features\' ORDER BY calculated_at', "time_series")], "percentunit")
    feature = panel("Feature evidence · distances are method-specific", "table", 12, 19, 12, 8,
        [sql_target(LATEST + " SELECT feature_name AS \"Feature\", metric_value AS \"Distance\", statistical_method AS \"Method\", threshold AS \"Threshold\", status AS \"Status\" FROM monitoring_metrics WHERE run_id=(SELECT run_id FROM latest) AND metric_name='feature_drift_score' ORDER BY feature_name")],
        description="Wasserstein is normalized numerical distance; Jensen–Shannon measures categorical shift. Scores are not pooled or treated as p-values.")
    feature["fieldConfig"]["defaults"]["decimals"] = 3
    feature["fieldConfig"]["overrides"] = [{"matcher": {"id": "byName", "options": "Status"}, "properties": [
        {"id": "mappings", "value": [{"type": "value", "options": {"drift": {"text": "DRIFT", "color": "red"}, "normal": {"text": "Normal", "color": "green"}}}]},
        {"id": "custom.cellOptions", "value": {"type": "color-text"}}]}]
    for name, width in [("Feature", 250), ("Distance", 85), ("Method", 150), ("Threshold", 90), ("Status", 90)]:
        feature["fieldConfig"]["overrides"].append({"matcher": {"id": "byName", "options": name}, "properties": [{"id": "custom.width", "value": width}]})
    panel("03 / Model performance · class 1 = machine failure", "row", 0, 27, 24, 1)
    for index, (metric, title) in enumerate([("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 score"), ("roc_auc", "ROC-AUC"), ("label_coverage", "Label coverage")]):
        panel(title, "stat", index*4, 28, 4, 4, [sql_target(LATEST + f" SELECT metric_value AS \"{title}\" FROM monitoring_metrics WHERE run_id=(SELECT run_id FROM latest) AND metric_name='{metric}'")], "percentunit",
              "Latest selected run only. Synthetic feature drift has no eligible ground truth; performance is unavailable, never filled from an older normal run.")
    panel("04 / Batch context & alert scope", "row", 0, 32, 24, 1)
    context = panel("Selected run · label count, mode and model identity", "table", 0, 33, 24, 5, [sql_target(LATEST + """
 SELECT calculated_at AS "Calculated", model_name AS "Model", model_version AS "Version",
 simulation_mode AS "Mode", batch_id AS "Batch", run_id AS "Run", sample_size AS "Predictions",
 (SELECT metric_value FROM monitoring_metrics labels WHERE labels.run_id=m.run_id AND labels.metric_name='labeled_predictions') AS "Eligible labels"
 FROM monitoring_metrics m WHERE run_id=(SELECT run_id FROM latest) AND metric_name='dataset_drift'
 """)])
    for name, width in [("Calculated", 160), ("Model", 270), ("Version", 75), ("Mode", 85), ("Batch", 280), ("Run", 280), ("Predictions", 100), ("Eligible labels", 120)]:
        context["fieldConfig"]["overrides"].append({"matcher": {"id": "byName", "options": name}, "properties": [{"id": "custom.width", "value": width}]})
    panel("Prediction request outcomes · since API start", "stat", 0, 38, 12, 4, [prom_target('sum by (status) (ai4i_prediction_requests_total{job="ai4i-api"})', '{{status}}', True)])
    p = panel("Grafana drift alert · latest result per model/version", "alertlist", 12, 38, 12, 4)
    p["options"] = {"viewMode": "list", "groupMode": "default", "maxItems": 10, "sortOrder": 1,
                    "stateFilter": {"firing": True, "pending": True, "normal": True, "noData": True, "error": True},
                    "alertName": "AI4I dataset drift"}
    p["description"] = "Alerts ignore dashboard mode/batch filters. Latest result per model/version across batches; null or stale data is No Data. Normal subsequent results recover."

    def variable(name, label, query, all_option=False):
        v = {"name": name, "label": label, "type": "query", "datasource": PG,
             "query": query, "definition": query, "refresh": 2, "multi": False,
             "includeAll": all_option, "options": [], "current": {}, "sort": 0}
        if all_option:
            v.update(allValue="__all", current={"text": "All", "value": "$__all"})
        return v
    variables = [
        variable("model", "Model", "SELECT DISTINCT model_name FROM monitoring_metrics WHERE run_id IS NOT NULL ORDER BY model_name"),
        variable("version", "Version", "SELECT DISTINCT model_version FROM monitoring_metrics WHERE run_id IS NOT NULL AND model_name=${model:sqlstring} ORDER BY model_version DESC"),
        variable("mode", "Simulation mode", "SELECT DISTINCT simulation_mode FROM monitoring_metrics WHERE run_id IS NOT NULL AND model_name=${model:sqlstring} AND model_version=${version:sqlstring} ORDER BY simulation_mode", True),
        variable("batch", "Batch", "SELECT DISTINCT batch_id AS __value, simulation_mode || ' · ' || left(batch_id,8) AS __text FROM monitoring_metrics WHERE run_id IS NOT NULL AND model_name=${model:sqlstring} AND model_version=${version:sqlstring} AND (${mode:sqlstring}='__all' OR simulation_mode=${mode:sqlstring}) AND $__timeFilter(calculated_at) ORDER BY __text", True),
    ]
    return {"uid": "ai4i-monitoring", "title": "AI4I MLOps Monitoring", "tags": ["AI4I", "MLOps"],
            "schemaVersion": 39, "version": 1, "editable": False, "timezone": "browser",
            "refresh": "5s", "time": {"from": "now-24h", "to": "now"},
            "templating": {"list": variables}, "panels": panels}


if __name__ == "__main__":
    (ROOT / "grafana/dashboards/ai4i-monitoring.json").write_text(json.dumps(build(), indent=2) + "\n")
