"""Deterministic synthetic daily data for a model incident investigation exercise."""

from datetime import date, timedelta

MODEL_NAME = "washing_machine_purchase_propensity"
FEATURE_NAME = "historical_email_clicks_90d"
START_DATE = date(2026, 9, 24)
NUMBER_OF_DAYS = 28
INCIDENT_START_INDEX = 24


def generate_raw_monitoring_rows() -> list[dict[str, str]]:
    """Return mock daily metrics with a late anomaly and two dirty input rows.

    The last four days simulate a silent upstream data-quality problem:
    a jump in missing feature values coincides with lower prediction scores.
    These metrics show correlation, not a proven root cause.

    Returns:
        Raw string-valued records ready for a Spark ingestion step.

    """
    records: list[dict[str, str]] = []

    for index in range(NUMBER_OF_DAYS):
        incident_period = index >= INCIDENT_START_INDEX
        null_rate = 0.52 + 0.015 * (index % 3) if incident_period else 0.018 + 0.002 * (index % 4)
        mean_feature = 0.9 + 0.1 * (index % 3) if incident_period else 3.4 + 0.1 * (index % 4)
        mean_score = 0.052 + 0.002 * (index % 3) if incident_period else 0.112 + 0.002 * (index % 4)

        records.append(
            {
                "snapshot_date": (START_DATE + timedelta(days=index)).isoformat(),
                "model_name": MODEL_NAME,
                "feature_name": FEATURE_NAME,
                "num_predictions": str(5000 + (index % 5) * 100),
                "feature_null_rate": f"{null_rate:.4f}",
                "feature_mean": f"{mean_feature:.4f}",
                "avg_prediction_score": f"{mean_score:.4f}",
                "pipeline_status": "SUCCESS",
            }
        )

    # Exact duplicate to demonstrate deduplication.
    records.append(records[3].copy())
    # Malformed extra row to demonstrate validation and rejection.
    records.append({**records[5], "num_predictions": "-100", "snapshot_date": "not-a-date"})
    return records
