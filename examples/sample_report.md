# LLM Cost Control Report

## Run Context
- Generated at: 2026-03-26T16:53:25.845423+00:00
- After log: `examples/sample_telemetry.jsonl`
- Mode: single-run summary

## Executive Summary
- Highest current cost route: `/conversation-turn`.
- Highest current error-burden route: `/answer-routed`.
- No before snapshot provided; comparison deltas were not computed.

## Telemetry Coverage Summary
- Current valid rows: 10
- Current malformed rows skipped: 0
- Current routes observed: 2
- Current successes: 8
- Current errors: 2

## Per-Route Aggregate Table
| Route | Request Count | Latency P50 (ms) | Latency P95 (ms) | Total Cost (USD) | Avg Cost (USD) | Error Rate | Schema-Valid Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| /answer-routed | 6 | 599.40 | 1788.25 | 0.008407 | 0.001401 | 0.1667 | 1.0000 |
| /conversation-turn | 4 | 1389.40 | 2362.50 | 0.009435 | 0.002359 | 0.2500 | 1.0000 |

## Eval Summary
### Classify Eval
Eval summary unavailable.

### Answer Routed Eval
Eval summary unavailable.

### Conversation Turn Eval
Eval summary unavailable.

## Pareto Analysis
### Highest Cost Routes
- `/conversation-turn` — total cost 0.009435 USD
- `/answer-routed` — total cost 0.008407 USD

### Highest Error Burden Routes
- `/answer-routed` — error count 1, error rate 0.1667
- `/conversation-turn` — error count 1, error rate 0.2500

## Recommendations
- Prioritize `/conversation-turn` first, because it currently dominates observed cost.
- Investigate `/answer-routed` first for failure reduction, because it currently has the highest observed error burden.
- Regression coverage is incomplete. Missing eval summaries for: Classify Eval, Answer Routed Eval, Conversation Turn Eval.
