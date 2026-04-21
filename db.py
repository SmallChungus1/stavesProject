import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from html import escape


DEFAULT_DB_PATH = "inference_logs.db"
TIME_RANGE_MAP = {
    "7d": "-7 days",
    "30d": "-30 days",
    "90d": "-90 days",
    "365d": "-365 days",
    "all": None,
}

PLOT_METRIC_COLUMNS = {
    "latency_ms": "latency_ms",
    "predicted_count": "predicted_count",
    "mean_confidence": "mean_confidence",
    "median_confidence": "median_confidence",
    "min_confidence": "min_confidence",
    "max_confidence": "max_confidence",
    "low_confidence_detections": "low_confidence_detections",
}


def get_db_path() -> str:
    return os.getenv("LOG_DB_PATH", DEFAULT_DB_PATH)


def ensure_parent_dir(db_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    ensure_parent_dir(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_log_db(db_path: str | None = None) -> str:
    path = db_path or get_db_path()
    with get_connection(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS inference_logs (
                inference_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                image_width INTEGER,
                image_height INTEGER,
                latency_ms REAL,
                predicted_count INTEGER,
                conf_threshold REAL,
                iou_threshold REAL,
                mean_confidence REAL,
                median_confidence REAL,
                min_confidence REAL,
                max_confidence REAL,
                low_confidence_detections INTEGER NOT NULL DEFAULT 0,
                localizer_version TEXT,
                counter_version TEXT,
                error_message TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_inference_logs_created_at
            ON inference_logs(created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_inference_logs_error_message
            ON inference_logs(error_message);
            """
        )
    return path


def insert_inference_log(record: dict, db_path: str | None = None) -> None:
    path = db_path or get_db_path()
    with get_connection(path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO inference_logs (
                inference_id,
                created_at,
                image_width,
                image_height,
                latency_ms,
                predicted_count,
                conf_threshold,
                iou_threshold,
                mean_confidence,
                median_confidence,
                min_confidence,
                max_confidence,
                low_confidence_detections,
                localizer_version,
                counter_version,
                error_message
            ) VALUES (
                :inference_id,
                :created_at,
                :image_width,
                :image_height,
                :latency_ms,
                :predicted_count,
                :conf_threshold,
                :iou_threshold,
                :mean_confidence,
                :median_confidence,
                :min_confidence,
                :max_confidence,
                :low_confidence_detections,
                :localizer_version,
                :counter_version,
                :error_message
            )
            """,
            record,
        )


def query_inference_logs(
    *,
    time_range: str = "7d",
    error_filter: str = "all",
    limit: int = 500,
    db_path: str | None = None,
) -> list[sqlite3.Row]:
    path = db_path or get_db_path()
    conditions = []
    params: list[object] = []

    range_modifier = TIME_RANGE_MAP.get(time_range, TIME_RANGE_MAP["7d"])
    if range_modifier is not None:
        conditions.append("created_at >= datetime('now', ?)")
        params.append(range_modifier)

    if error_filter == "errors_only":
        conditions.append("error_message IS NOT NULL AND TRIM(error_message) <> ''")
    elif error_filter == "success_only":
        conditions.append("(error_message IS NULL OR TRIM(error_message) = '')")

    where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    with get_connection(path) as conn:
        return conn.execute(
            f"""
            SELECT
                inference_id,
                created_at,
                image_width,
                image_height,
                latency_ms,
                predicted_count,
                conf_threshold,
                iou_threshold,
                mean_confidence,
                median_confidence,
                min_confidence,
                max_confidence,
                low_confidence_detections,
                localizer_version,
                counter_version,
                error_message
            FROM inference_logs
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()


def build_plot_series(
    rows: list[sqlite3.Row],
    *,
    metric: str,
) -> dict[str, object]:
    column = PLOT_METRIC_COLUMNS.get(metric)
    ordered_rows = sorted(rows, key=lambda row: row["created_at"])

    if metric == "error_rate_pct":
        return {
            "mode": "daily_aggregated",
            "raw_points": [],
            "aggregate_points": _build_error_rate_series(ordered_rows),
            "aggregate_label": "Daily Error Rate",
        }
    if column is None:
        raise ValueError(f"Unsupported plot metric '{metric}'")

    raw_points = []
    for row in ordered_rows:
        value = row[column]
        if value is None:
            continue
        raw_points.append({"timestamp": row["created_at"], "value": value})

    return {
        "mode": "raw_plus_daily_median",
        "raw_points": raw_points,
        "aggregate_points": _build_metric_aggregate_series(raw_points),
        "aggregate_label": "Daily Median",
    }


def _parse_sql_timestamp(timestamp: str) -> datetime:
    return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")


def _bucket_timestamp(timestamp: str) -> str:
    dt = _parse_sql_timestamp(timestamp)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")


def _build_metric_aggregate_series(points: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[str, list[float]] = defaultdict(list)

    for point in points:
        bucket = _bucket_timestamp(str(point["timestamp"]))
        buckets[bucket].append(float(point["value"]))

    aggregated = []
    for bucket in sorted(buckets):
        values = sorted(buckets[bucket])
        midpoint = len(values) // 2
        if len(values) % 2 == 0:
            median = (values[midpoint - 1] + values[midpoint]) / 2
        else:
            median = values[midpoint]
        aggregated.append({"timestamp": bucket, "value": round(median, 4)})

    return aggregated


def _build_error_rate_series(
    rows: list[sqlite3.Row],
) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"total_requests": 0, "error_requests": 0})

    for row in rows:
        bucket = _bucket_timestamp(row["created_at"])
        bucket_entry = buckets[bucket]
        bucket_entry["total_requests"] += 1
        if row["error_message"] is not None and str(row["error_message"]).strip():
            bucket_entry["error_requests"] += 1

    points = []
    for bucket in sorted(buckets):
        total_requests = buckets[bucket]["total_requests"]
        error_requests = buckets[bucket]["error_requests"]
        error_rate_pct = round((100.0 * error_requests / total_requests), 4) if total_requests else 0.0
        points.append(
            {
                "timestamp": bucket,
                "value": error_rate_pct,
                "total_requests": total_requests,
                "error_requests": error_requests,
            }
        )

    return points


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return escape(str(value))


def render_logs_table(rows: list[sqlite3.Row]) -> str:
    if not rows:
        return "<p class='table-empty'>No inference logs found for this filter.</p>"

    columns = [
        ("created_at", "Timestamp"),
        ("inference_id", "Inference ID"),
        ("image_width", "Width"),
        ("image_height", "Height"),
        ("latency_ms", "Latency (ms)"),
        ("predicted_count", "Predicted Count"),
        ("conf_threshold", "Conf Thresh"),
        ("iou_threshold", "IoU Thresh"),
        ("mean_confidence", "Mean Conf"),
        ("median_confidence", "Median Conf"),
        ("min_confidence", "Min Conf"),
        ("max_confidence", "Max Conf"),
        ("low_confidence_detections", "Low Conf Count"),
        ("localizer_version", "Localizer Ver"),
        ("counter_version", "Counter Ver"),
        ("error_message", "Error"),
    ]

    head = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{_format_cell(row[key])}</td>" for key, _ in columns)
        body_rows.append(f"<tr>{cells}</tr>")

    return (
        "<table>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def utc_now_sql() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    db_path = init_log_db()
    print(f"Initialized log database at {db_path}")
