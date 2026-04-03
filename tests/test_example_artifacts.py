"""Tests for example artifacts integrity and parseability.

These tests validate that the example telemetry and report artifacts remain
parseable and can be consumed by the reporting pipeline without error.
"""

import json
from pathlib import Path

import pytest

from reporting.make_report import (
    build_route_aggregates,
    load_jsonl_telemetry,
    render_markdown_report,
)
from turnpike.gateway.cost_model import estimate_cost

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_TELEMETRY_PATH = PROJECT_ROOT / "examples" / "sample_telemetry.jsonl"
SAMPLE_REPORT_PATH = PROJECT_ROOT / "examples" / "sample_report.md"


def test_sample_telemetry_exists_and_parseable():
    """Validate sample_telemetry.jsonl exists and all lines parse as JSON."""
    assert SAMPLE_TELEMETRY_PATH.exists(), "sample_telemetry.jsonl not found"

    lines = SAMPLE_TELEMETRY_PATH.read_text().strip().split("\n")
    assert len(lines) == 10, f"Expected 10 lines, got {len(lines)}"

    for i, line in enumerate(lines):
        event = json.loads(line)
        assert isinstance(event, dict), f"Line {i}: not a dict"


def test_sample_telemetry_required_fields():
    """Validate each event has required fields with correct types."""
    lines = SAMPLE_TELEMETRY_PATH.read_text().strip().split("\n")

    required_fields = [
        "route",
        "latency_ms",
        "estimated_cost_usd",
        "status",
        "schema_valid",
        "model_selected",
        "routing_decision",
        "schema_version",
    ]

    for i, line in enumerate(lines):
        event = json.loads(line)

        for field in required_fields:
            assert field in event, f"Line {i}: missing field '{field}'"

        assert event["status"] in (
            "success",
            "error",
        ), f"Line {i}: invalid status '{event['status']}'"
        assert event["route"].startswith("/"), f"Line {i}: route must start with '/'"


def test_sample_telemetry_cost_correctness():
    """Validate cost calculations match the pricing model exactly."""
    lines = SAMPLE_TELEMETRY_PATH.read_text().strip().split("\n")

    for i, line in enumerate(lines):
        event = json.loads(line)
        model = event["model_selected"]
        tokens_in = event["tokens_in"]
        tokens_out = event["tokens_out"]
        reported_cost = event["estimated_cost_usd"]

        if event["status"] == "success":
            expected_cost = estimate_cost(model, tokens_in, tokens_out)
            assert reported_cost == pytest.approx(expected_cost, abs=1e-7), (
                f"Line {i}: cost mismatch"
            )
        else:
            assert reported_cost == 0.0, f"Line {i}: error event should have zero cost"


def test_sample_telemetry_normalizes_for_reporting():
    """Validate telemetry normalizes correctly for the reporting pipeline."""
    normalized_rows, malformed_count = load_jsonl_telemetry(str(SAMPLE_TELEMETRY_PATH))

    assert len(normalized_rows) == 10, "Expected 10 normalized rows"
    assert malformed_count == 0, f"Expected 0 malformed rows, got {malformed_count}"


def test_sample_report_exists():
    """Validate sample_report.md exists and contains expected sections."""
    assert SAMPLE_REPORT_PATH.exists(), "sample_report.md not found"

    content = SAMPLE_REPORT_PATH.read_text()
    assert "# LLM Cost Control Report" in content, "Missing report title"
    assert "Per-Route Aggregate Table" in content, "Missing aggregate table section"


def test_report_generator_produces_output():
    """Validate report generator can consume sample telemetry and produce output."""
    # Load telemetry
    normalized_rows, malformed_count = load_jsonl_telemetry(str(SAMPLE_TELEMETRY_PATH))
    assert len(normalized_rows) > 0, "No normalized rows loaded"

    # Build aggregates
    route_aggregates, overall_aggregate = build_route_aggregates(normalized_rows)
    assert len(route_aggregates) > 0, "No route aggregates built"

    # Render report
    report_output = render_markdown_report(
        before_log_path=None,
        after_log_path=str(SAMPLE_TELEMETRY_PATH),
        before_rows=[],
        after_rows=normalized_rows,
        malformed_before_count=0,
        malformed_after_count=malformed_count,
        before_by_route=None,
        after_by_route=route_aggregates,
        before_overall=None,
        after_overall=overall_aggregate,
        eval_payloads={},
    )

    assert isinstance(report_output, str), "Report output should be a string"
    assert len(report_output) > 0, "Report output should not be empty"
    assert "LLM Cost Control Report" in report_output, "Missing report title"
    assert "/answer-routed" in report_output, "Missing /answer-routed route"
    assert "/conversation-turn" in report_output, "Missing /conversation-turn route"
