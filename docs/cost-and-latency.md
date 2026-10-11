# Cost and latency

`scripts/cost_report.py` reads `traces/<case_id>.jsonl`. Each span's `prompt_tokens` and `completion_tokens` are Ollama's `prompt_eval_count` and `eval_count`. A completed case has four spans and every status is `ok`. The averages use only those cases. Every other file is listed and left out.

## Setup

The model is `qwen3.5:9b` on local Ollama. Hardware: Apple Silicon Mac (aarch64); the Docker VM reports 10 CPUs and 7.7 GiB RAM. The pipeline runs in a Docker Linux container. Ollama runs on the Mac host. Timeout is 30 s, with 3 risk-scoring attempts and a 1 s then 2 s backoff.

## Measured results

```
Completed cases: 5

Per-agent averages (completed cases only)
agent           prompt  completion  total   ms
--------------  ------  ----------  ------  -------
intake          1221.0  131.8       1352.8  8715.0
enrichment      2826.8  126.8       2953.6  8074.6
risk_scoring    1218.0  203.8       1421.8  10953.0
recommendation  1370.6  61.0        1431.6  4298.3

Per case (completed traces only)
  average tokens: 7159.8
  maximum tokens: 7465
  average seconds: 32.04
  maximum seconds: 42.45

Local cost: $0

Hosted cost
  claude-haiku-4.5: $0.009253 per case, $9.2534 per 1,000 cases
  claude-sonnet-4.6: $0.027760 per case, $27.7602 per 1,000 cases

Worst-case latency
  timeout: 30s (DEFAULT_TIMEOUT_SECONDS)
  backoff: 3s
  normal case: 32.04s (average completed case)
  worst case: 125.04s (normal case + 3 timeouts + backoff)

Cases with no completed trace (not averaged)
  dob_mismatch
  example
  incomplete
  missing_vehicle_model
  unknown_person
```

| agent | prompt | completion | total | ms |
| --- | --- | --- | --- | --- |
| intake | 1221.0 | 131.8 | 1352.8 | 8715.0 |
| enrichment | 2826.8 | 126.8 | 2953.6 | 8074.6 |
| risk_scoring | 1218.0 | 203.8 | 1421.8 | 10953.0 |
| recommendation | 1370.6 | 61.0 | 1431.6 | 4298.3 |

| | tokens | seconds |
| --- | --- | --- |
| average | 7159.8 | 32.04 |
| maximum | 7465 | 42.45 |

## Cost

Local cost is $0 per token, and the hosted rows show what the same measured tokens would cost on a cloud model. `hosted_cost` is `(prompt_tokens / 1_000_000) * input + (completion_tokens / 1_000_000) * output`. A `None` rate prints "rate not set".

| model | per case | per 1,000 cases |
| --- | --- | --- |
| local Ollama | $0 | $0 |
| Claude Haiku 4.5 | $0.009253 | $9.2534 |
| Claude Sonnet 4.6 | $0.027760 | $27.7602 |

## Latency

The normal case is the report's average, 32.04 s, and its maximum, 42.45 s.

The worst case is 125.04 s (about 125 s): the normal run plus 3 × 30 s timeouts plus 3 s of backoff, `32.04 + 3 × 30 + 3`. That figure is an upper bound, because on that path recommendation never runs.

Worst-case latency, in milliseconds, is the normal case duration plus three client timeouts plus the backoff:

`normal_case_ms + timeouts * timeout_s * 1000 + backoff_s * 1000`

`timeouts` is 3 and `timeout_s` is 30 (`DEFAULT_TIMEOUT_SECONDS` in `src/underwriting/llm_adapter.py`). `backoff_s` is 3, the sum of `[1, 2]`. That sum is the path where risk scoring times out three times and the case escalates.

## Throughput

At the average of 32.04 seconds per case, one machine running cases one at a time handles 3600 / 32.04 = 112.36 cases per hour. A 10-hour overnight window is 10 × (3600 / 32.04) = 1123.60 cases.

## Verdict

This is not suitable for a real-time quote, because a customer would wait about half a minute, and up to about 2 minutes in the worst case. It is suitable for overnight batch, or an asynchronous "decision by email" flow: the same machine finishes 112.36 cases an hour and 1123.60 cases in that 10-hour window. A faster hosted model, running agents or lookups in parallel, or a smaller prompt would change this.
