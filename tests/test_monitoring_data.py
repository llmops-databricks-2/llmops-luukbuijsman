"""Unit tests for Lecture 1 synthetic monitoring data."""

from datetime import date

from arxiv_curator.monitoring_data import NUMBER_OF_DAYS, generate_raw_monitoring_rows


def test_raw_data_has_expected_good_rows_and_dirty_examples() -> None:
    """Ensure the lesson dataset is small, repeatable, and contains cleaning cases."""
    rows = generate_raw_monitoring_rows()
    assert len(rows) == NUMBER_OF_DAYS + 2
    assert rows == generate_raw_monitoring_rows()
    assert rows[3] == rows[NUMBER_OF_DAYS]
    assert rows[-1]["snapshot_date"] == "not-a-date"
    assert date.fromisoformat(rows[0]["snapshot_date"]) < date.fromisoformat(rows[27]["snapshot_date"])


def test_incident_is_visible_in_daily_metrics() -> None:
    """Verify the synthetic incident will be detectible by later analysis."""
    rows = generate_raw_monitoring_rows()
    baseline, incident = rows[23], rows[24]
    assert float(incident["feature_null_rate"]) > float(baseline["feature_null_rate"]) + 0.4
    assert float(incident["avg_prediction_score"]) < float(baseline["avg_prediction_score"]) - 0.04
