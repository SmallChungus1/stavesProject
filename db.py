import os
import sqlite3
from datetime import datetime, timezone
from html import escape


DEFAULT_DB_PATH = "inference_logs.db"
TIME_RANGE_MAP = {
    "24h": "-1 day",
    "7d": "-7 days",
    "30d": "-30 days",
    "90d": "-90 days",
    "365d": "-365 days",
    "all": None,
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
    time_range: str = "24h",
    error_filter: str = "all",
    limit: int = 500,
    db_path: str | None = None,
) -> list[sqlite3.Row]:
    path = db_path or get_db_path()
    conditions = []
    params: list[object] = []

    range_modifier = TIME_RANGE_MAP.get(time_range, TIME_RANGE_MAP["24h"])
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
