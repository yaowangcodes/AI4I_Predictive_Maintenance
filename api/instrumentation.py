"""Process-local HTTP metrics with bounded labels and isolated app registries."""
from time import perf_counter

from prometheus_client import CollectorRegistry, Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response


class HTTPMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        labels = ["method", "endpoint", "status"]
        self.requests = Counter("ai4i_http_requests_total", "Completed HTTP requests", labels, registry=self.registry)
        self.errors = Counter("ai4i_http_errors_total", "HTTP responses with status >= 400", labels, registry=self.registry)
        self.duration = Histogram("ai4i_http_request_duration_seconds", "Full HTTP response duration", labels,
                                  buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10), registry=self.registry)
        self.predictions = Counter("ai4i_prediction_requests_total", "POST /predict responses", ["status"], registry=self.registry)

    def response(self):
        return Response(generate_latest(self.registry), headers={"Content-Type": CONTENT_TYPE_LATEST})


class MetricsMiddleware:
    def __init__(self, app, metrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].rstrip("/") == "/metrics":
            return await self.app(scope, receive, send)
        start = perf_counter()
        status = 500

        async def capture(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, capture)
        finally:
            route = scope.get("route")
            endpoint = getattr(route, "path", "unmatched")
            method = scope["method"]
            if method not in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}:
                method = "OTHER"
            labels = (method, endpoint, str(status))
            self.metrics.requests.labels(*labels).inc()
            self.metrics.duration.labels(*labels).observe(perf_counter() - start)
            if status >= 400:
                self.metrics.errors.labels(*labels).inc()
            if method == "POST" and endpoint == "/predict":
                self.metrics.predictions.labels(str(status)).inc()


def instrument_app(app):
    if hasattr(app.state, "http_metrics"):
        return app.state.http_metrics
    metrics = HTTPMetrics()
    app.state.http_metrics = metrics
    app.add_middleware(MetricsMiddleware, metrics=metrics)
    app.add_api_route("/metrics", metrics.response, methods=["GET"], include_in_schema=False)
    return metrics
