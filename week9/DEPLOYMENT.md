# Week 9: Deploying the Risk Scoring Agent

One agent from the week 8 underwriting pipeline, the **risk scoring agent**, deployed two ways:

- **AWS (simulated):** a container on ECS Fargate in us-east-1, calling **Gemma 3 27B on Amazon Bedrock**. It runs under a dedicated least-privilege IAM role, in private subnets that reach AWS only through VPC endpoints.
- **On-prem (simulated):** the same agent calling **qwen3.5:9b on local Ollama**, so the model runs inside our own boundary.

> **Simulation note.** The instructor's update for this lab was: do not use any cloud service (AWS, Azure, or GCP); simulate it. The AWS deployment is therefore written as real configuration that was never run: IAM policies, network policies, the ECS task definition, and a deployment script. The on-prem deployment did run, as the week 8 pipeline on local Ollama, and its results are in `traces/`. Everything that needs a real AWS account to confirm is listed under [What was not verified](#what-was-not-verified).

### At a glance

| | AWS (simulated) | On-prem (simulated) |
|---|---|---|
| Model | Gemma 3 27B, run by AWS (Bedrock, us-east-1) | qwen3.5:9b, run by us (Ollama on our machine) |
| Agent runs in | ECS Fargate task (ARM64), private subnets | Docker container on the Mac, Ollama on the host |
| Code reaches the model through | Bedrock Converse API, via a VPC endpoint | Ollama HTTP API at `host.docker.internal:11434` |
| Credentials | Temporary, from the ECS task role. No keys anywhere | None needed |
| Data leaves our boundary | Yes, to Bedrock, but only within us-east-1 | No |
| Cost per request at 10,000 a day | About $0.0007 all-in (see [Cost](#cost-at-10000-requests-a-day)) | $0 per token; hardware, power, and people instead |
| Status | Designed and written, never run | Ran all 10 cases |

## Which agent, and what it is given

**The risk scoring agent was chosen** for four reasons:

- **Self-contained.** It takes one message (`EnrichedCase`), makes one model call, and returns one message (`RiskAssessment`).
- **Easy to compare.** Its output is a number (a 0–100 score and a tier), so the two deployments can be compared directly.
- **Same rules everywhere.** Code around the model enforces the rules regardless of which model runs: a minimum score of 30 when there is a discrepancy flag or a data gap, and the tier from fixed score ranges.
- **Already handles timeouts.** It already has the week 8 retry on timeout.

**Test inputs.** In the pipeline, risk scoring receives an `EnrichedCase` from the enrichment agent and never sees the application text. In the cloud, only the risk agent is deployed, so its input has to be supplied. [`build_fixtures.py`](build_fixtures.py) ran the week 8 intake and enrichment agents once on local Ollama and saved the five resulting `EnrichedCase` messages in [`fixtures/`](fixtures/). Each one was checked against the enrichment output in `traces/` (all five match), so both deployments score exactly the input the on-prem run scored.

Only five of the nine test cases in `data/manifest.json` reach risk scoring. The other four are stopped earlier by design, so they never reach the deployed agent:

| Case | Stopped by | Outcome |
|---|---|---|
| `unknown_person` | Enrichment | Escalated: insufficient information found |
| `dob_mismatch` | Enrichment | Escalated: possible identity mismatch |
| `incomplete` | Intake | Rejected: date of birth is missing |
| `missing_vehicle_model` | Intake | Escalated: vehicle model is missing |

## Part 1: AWS deployment (simulated)

### Architecture

```
 Deployer laptop ── docker push ──▶ ECR: risk-scoring-agent:v1 (immutable tag, scanned on push)

 VPC 10.0.0.0/16: no internet gateway, no NAT, route table has only the local route
 ┌────────────────────────────────────────────────────────────────────────┐
 │  private subnet us-east-1a              private subnet us-east-1b       │
 │  ┌──────────────────────────────────┐                                  │
 │  │ Fargate task, ARM64, 0.25 vCPU   │  task role: bedrock:InvokeModel  │
 │  │ risk-scoring-agent               │  on Gemma 3 27B only             │
 │  │ SG: no inbound; HTTPS out only   │                                  │
 │  │     to the endpoints and S3      │                                  │
 │  └────────────────┬─────────────────┘                                  │
 │                   │ HTTPS 443                                          │
 │  interface endpoints (SG: HTTPS in from the task only)                 │
 │      ecr.api      ecr.dkr      logs      bedrock-runtime               │
 │  gateway endpoint: s3 (ECR image-layer bucket only)                    │
 └───────┬──────────────┬──────────────────┬──────────────────────────────┘
         ▼              ▼                  ▼
   ECR (image)   CloudWatch Logs    Bedrock: Gemma 3 27B, us-east-1
```

### Model: Gemma 3 27B on Amazon Bedrock

- **Why not Claude:** Claude models are not allowed during the training period, so the agent uses Google's open-weight Gemma 3 27B (`google.gemma-3-27b-it`).
- **Requests stay in one Region.** Bedrock lists no cross-region (geo or global) inference profiles for Gemma 3 27B, only in-Region access. A request sent to us-east-1 is processed in us-east-1. We use us-east-1 because that is our Region.
- **India option.** The same deployment in **ap-south-1 (Mumbai)**, where Gemma 3 27B is also offered in-Region, would keep every request inside India. The change is configuration only: `AWS_REGION`, the model ARN in the task role and endpoint policies, and the endpoint service names.
- **No AWS Marketplace step.** Google models are not sold through AWS Marketplace, so no `aws-marketplace:*` permissions are needed to start using the model.
- **Limits:** 128K-token context, 8K max output tokens.
- **Lifecycle:** the model is in the Legacy state, with end of life on **30 March 2027**. See [Model lifecycle](#model-lifecycle-end-of-life-alerts) for the alerts.

### Code: the Bedrock adapter

The week 8 agents talk to a model only through one interface, `LLMAdapter.complete(system, user) -> LLMResponse`. [`risk_agent/bedrock_adapter.py`](risk_agent/bedrock_adapter.py) adds a `BedrockAdapter` that implements it. The agent, prompt, minimum score, tier, and retry are unchanged; the only difference between the deployments is which adapter is passed in.

**Converse API.** Bedrock has two ways to call a model:

- `InvokeModel` needs a request body written for each model family.
- `Converse` uses **one request shape for every model**.

The adapter uses Converse. Replacing Gemma 3 27B at end of life (for example with Gemma 4) is a change to `BEDROCK_MODEL_ID` and the IAM ARNs. No code changes.

**Checking that the reply is JSON.** Ollama has a `format="json"` parameter that makes the model return JSON. The Converse API has no such parameter. Every reply is therefore checked by the week 8 [`_llm_json.py`](../src/underwriting/agents/_llm_json.py):

1. It removes markdown code fences.
2. It extracts the JSON object.
3. It validates the object against the agent's schema.
4. If validation fails, it makes one repair call.
5. If the repair also fails, the case escalates to **refer**.

Bedrock's structured outputs feature, which this model supports, is a possible later improvement.

| Choice | Why |
|---|---|
| Temperature 0 | Repeatable scores, same as the Ollama adapter |
| `maxTokens` 512 | Replies are 199–209 tokens (5 factors). The cap is a ceiling, not a charge, so 512 leaves headroom at no extra cost. A cut-off reply would cost a repair call |
| `BEDROCK_SYSTEM_IN_USER` switch | Gemma has no native system role. If Bedrock rejects a separate system prompt for it, setting this to 1 places the rules at the top of the user message: a configuration change, no rebuild |
| 30 s read timeout, 5 s connect timeout | Same timeout as Ollama. Timeouts become `LLMTimeoutError`, so the week 8 retry (3 attempts, waiting 1 s then 2 s) applies unchanged |
| botocore's own retries turned off | Otherwise each of our 3 attempts could be retried again inside boto3: up to 9 attempts and about 270 s worst case instead of about 90 s. Throttling is then not retried automatically, so the Bedrock quota must cover peak load |
| No keys | boto3's default credential chain picks up temporary task role credentials. The model card's API-key example is deliberately not used |
| Region and model from environment variables | Moving to ap-south-1 or to a new model is configuration |
| `inputTokens` / `outputTokens` mapped to the week 8 token fields | Cost tracking works unchanged |

**What Gemma sees.** The agent sends Gemma only the slice it needs: the four bands, discrepancy flags, data gaps, summary, vehicle, and coverage. **The applicant's name and date of birth never reach the model**, and a test asserts it.

**Entrypoint.** [`risk_agent/run_risk_agent.py`](risk_agent/run_risk_agent.py) is what the container runs:

- **Input:** it validates every input file before calling the model.
- **Scoring:** it runs the agent with the week 8 retry.
- **Output:** one JSON line per case on stdout, which ECS sends to CloudWatch Logs. It writes nothing to disk.
- **Exit codes:** 0 when every case was scored, 1 when a case escalated, 2 when an input file is missing or invalid.
- **Other errors:** errors such as `AccessDeniedException` are deliberately not caught. The traceback reaches the logs, and the task stops with a failure code, so a permission mistake is visible at once.

**Tests.** [`tests/`](tests/) has 16 tests for the adapter and the entrypoint. They use fake clients, need no AWS, and run in CI with the week 8 suite.

### Container image

[`Dockerfile`](Dockerfile) and [`Dockerfile.dockerignore`](Dockerfile.dockerignore):

| Decision | Why |
|---|---|
| Built from the repo root with `-f week9/Dockerfile` | The image needs `src/underwriting` and `uv.lock`; the repo root is not modified |
| Allow-list ignore file | Only 6 paths reach the build. `data/` (applicant databases), `tests/`, `traces/`, `.venv`, and `.git` never do |
| Two stages | The build stage installs dependencies with uv. The runtime stage gets only the finished environment, with no build tools |
| `uv sync --locked --no-dev` | The exact versions week 8 CI tested, and no pytest, ruff, or mypy |
| `boto3==1.43.114`, `botocore==1.43.114` pinned in the Dockerfile | `pyproject.toml` stays unchanged. Both are pinned because boto3 alone allows a range of botocore versions |
| `python:3.12.14-slim-trixie`, `uv:0.12.12` | Exact, small, reproducible |
| Contents: `src/underwriting`, `risk_agent`, `fixtures` | The whole package, because `pipeline/__init__.py` imports the full chain. Only risk scoring runs |
| Non-root user 10001 | A compromised process is not root |
| `PYTHONDONTWRITEBYTECODE`, nothing written to disk | The task runs with a read-only root filesystem |
| `PYTHONUNBUFFERED` | Logs reach CloudWatch immediately |
| `ENTRYPOINT` fixed, `CMD` lists the 5 fixtures | The task can score one case by overriding the command, with no rebuild |
| ARM64 | Fargate on Graviton costs about 20% less. The task definition must also say ARM64, or the task fails with `exec format error` |
| No secrets, no exposed ports | Nothing listens, so the security group needs no inbound rules |

The fixtures contain (fictional) applicant data baked into the image. That is acceptable for a demo only. In production the input would arrive from the enrichment step through a queue.

### IAM: two roles, each with the minimum

ECS uses two separate roles, and each gets its own customer managed policy:

- **Execution role (`risk-agent-execution-role`):** used by **ECS itself** to start the container.
- **Task role (`risk-agent-task-role`):** what **our code** runs as. This is the dedicated role the lab asks for.

Both are created and checked **before** anything uses them. This is step 1 of [`aws/deploy_steps.sh`](aws/deploy_steps.sh).

**Execution role:** [`aws/iam/execution-role-policy.json`](aws/iam/execution-role-policy.json)

| Permission | Limited to | Why it is needed |
|---|---|---|
| `ecr:GetAuthorizationToken` | `*` | Log in to ECR. AWS does not allow this action to be scoped to a repository |
| `ecr:BatchCheckLayerAvailability`, `ecr:GetDownloadUrlForLayer`, `ecr:BatchGetImage` | `repository/risk-scoring-agent` | Pull our image and no other image |
| `logs:CreateLogStream`, `logs:PutLogEvents` | `log-group:/ecs/risk-scoring-agent:*` | Write our logs. It cannot create log groups; ours is created in advance |

AWS's managed `AmazonECSTaskExecutionRolePolicy` grants these permissions on every repository and every log group. We wrote our own policy instead.

**Task role:** [`aws/iam/task-role-policy.json`](aws/iam/task-role-policy.json)

| Permission | Limited to | Why it is needed |
|---|---|---|
| `bedrock:InvokeModel` | `arn:aws:bedrock:us-east-1::foundation-model/google.gemma-3-27b-it` | The Converse API is authorized by this action. One model, one Region |

That is the whole policy. The task role has:

- no access to any other model, not even Gemma 3 12B
- no streaming, because the agent does not stream
- no S3, Secrets Manager, or any other service

**Trust policies** (who may assume each role):

- **Task role:** only `ecs-tasks.amazonaws.com`, and only for tasks in our account (`aws:SourceAccount` and `aws:SourceArn`). This is AWS's recommended guard against the "confused deputy" problem, where another account tricks a service into using our role.
- **Execution role:** AWS's documented trust policy, which has no conditions.

**Proving least privilege.** After creating the roles, the script checks three cases with `aws iam simulate-principal-policy`:

| Action | Resource | Expected |
|---|---|---|
| `bedrock:InvokeModel` | Gemma 3 27B | `allowed` |
| `bedrock:InvokeModel` | Gemma 3 12B | `implicitDeny` |
| `s3:GetObject` | any bucket | `implicitDeny` |

### Networking

| Piece | Setting | Reasoning |
|---|---|---|
| Subnets | Two **private** subnets, 10.0.1.0/24 (us-east-1a) and 10.0.2.0/24 (us-east-1b). `assignPublicIp=DISABLED` | Nothing calls the agent. It starts, makes outbound calls, and stops. Two AZs let it run if one AZ has problems |
| Route table | Only the local route. No internet gateway, no NAT | Nothing in the subnets can reach the internet |
| Task security group | **No inbound rules.** The default allow-all outbound rule is removed. Outbound allows only HTTPS 443 to the endpoint security group and to the S3 prefix list | The container listens on no port. It may reach only the AWS services it needs |
| Endpoint security group | Inbound HTTPS 443 **only from the task security group**. No outbound rules | Nothing else in the VPC can use the endpoints |
| Interface endpoints | `ecr.api`, `ecr.dkr` (pull the image), `logs` (CloudWatch), `bedrock-runtime` (Gemma), all with private DNS | AWS services are reached over private IPs inside the VPC. Traffic never crosses the public internet |
| Bedrock endpoint policy | Only `risk-agent-task-role`, only `bedrock:InvokeModel`, only Gemma 3 27B | A second lock: even a misconfigured role cannot use another model through this endpoint |
| S3 gateway endpoint (free) | Policy allows only `s3:GetObject` on `prod-us-east-1-starport-layer-bucket`, where ECR stores image layers | The task cannot upload data to any other bucket, which closes a common data-leak path |

Two things need no extra configuration:

- **No STS endpoint.** Fargate hands the task role's credentials to the container from a local ECS address.
- **No DNS rule.** Security groups do not filter traffic to the VPC's DNS resolver.

**Alternatives considered:**

| Option | Verdict |
|---|---|
| Public subnet with a public IP, no inbound rules | Cheapest, but the task gets a route to the whole internet, and Bedrock traffic goes to the public endpoint. Rejected |
| Private subnet with a NAT gateway | Works, but it also opens a route to the internet. It costs $2.16 a day for two NAT gateways, plus data charges. Rejected |
| **Private subnet with VPC endpoints** | **Chosen.** No internet route at all, and four endpoints ($1.92 a day) cost less than two NAT gateways |

### Deploy and verify, step by step

[`aws/deploy_steps.sh`](aws/deploy_steps.sh) holds every command in order. It refuses to run unless `CONFIRM_REAL_AWS=yes` is set.

1. **IAM.** Create both roles and policies, then confirm them with `get-role`, `list-attached-role-policies`, and `simulate-principal-policy`.
2. **ECR.** Create a private repository with immutable tags and scanning on push. Build for `linux/arm64`, push `v1`, and read the scan findings.
3. **Network.** Check which AZs offer `bedrock-runtime`. Then create the VPC, subnets, route table, both security groups and their rules, and the five endpoints.
4. **Logs.** Create `/ecs/risk-scoring-agent` with 30-day retention.
5. **ECS.** Create the cluster and register [`aws/ecs/task-definition.json`](aws/ecs/task-definition.json):
   - Fargate, ARM64, 0.25 vCPU, 512 MB
   - both roles set explicitly
   - runs as user 10001, with a read-only root filesystem and all Linux capabilities dropped

   Then run the task once in the private subnets.
6. **Confirm.** Wait for the task to stop. **Exit code 0** means every case was scored. Then read the results with `aws logs tail`. Each case is one line in this format:

   ```json
   {"case_id": "tyler_brandt", "model": "google.gemma-3-27b-it", "status": "scored", "score": 0, "tier": "...", "factors": [], "attempts": 1, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 1, "duration_ms": 0}
   ```

   (The values here are placeholders; the real ones would come from Gemma.)
7. **Teardown.** `deploy_steps.sh teardown` deletes everything in reverse order, finding resources by their `Project` tag.

**Optional hardening, not included.** Three further restrictions are possible:

- Restrict the task role to calls that come through our Bedrock endpoint (`aws:SourceVpce`), so stolen credentials would be useless outside the VPC. It is left out because the endpoint ID exists only after the network is built, which conflicts with creating IAM first.
- Add the same kind of condition to the execution role's ECR permissions.
- Pin the base image by digest (`@sha256:...`) instead of by tag.

## Part 2: On-prem deployment (simulated)

**What runs.** The week 8 pipeline runs unchanged:

- `scripts/run_cases.py` runs in a Docker container on a Mac.
- Ollama runs on the host with **qwen3.5:9b**, and the container reaches it at `http://host.docker.internal:11434`.
- The agent calls it through the week 8 `OllamaAdapter`: `format="json"`, temperature 0, thinking off, 30 s timeout.

The model already runs inside our own boundary, so **no new code was needed** for this deployment. Running week 8 *is* the on-prem deployment.

**Same input.** The risk scoring step in these runs received exactly the values saved as fixtures for the AWS deployment.

**Results** (from `traces/`, run on 16 Oct 2026):

| Case | Score | Tier | Expected tier (manifest) | Final decision | Acceptable (manifest) | Risk scoring time | Tokens in / out |
|---|---:|---|---|---|---|---:|---|
| `maria_ortiz` | 15 | low | low ✓ | approve | approve ✓ | 10.5 s | 1,203 / 204 |
| `tyler_brandt` | 92 | severe | high or severe ✓ | deny | deny or refer, never approve ✓ | 10.3 s | 1,226 / 199 |
| `clean_claim_mismatch` | 76 | high | moderate, high, or severe ✓ | refer | refer ✓ | 12.4 s | 1,259 / 209 |
| `messy_formats` | 28 | low | not set | approve | approve or refer ✓ | 11.8 s | 1,196 / 203 |
| `timeout_case` | 24 | low | not set | approve | approve or refer ✓ | 10.7 s | 1,203 / 204 |

- **All five match the manifest.**
  - Tyler's application tells the model to approve; the result was deny.
  - Priya's clean-record claim was flagged as a discrepancy, so the case was referred.
- **The four early-stop cases** stopped exactly as listed in [Which agent](#which-agent-and-what-it-is-given).
- **Speed.** Risk scoring averaged **11.1 s per case**, and a full case took 27–43 s.
- **Capacity.** One machine scoring one case at a time manages about **7,750 risk scores a day** (86,400 s ÷ 11.1 s). That is short of 10,000, so this volume would need at least two machines like it, or a GPU server.

## Part 3: What changed between AWS and on-prem

| Concern | AWS | On-prem / air-gapped |
|---|---|---|
| Who runs the model | AWS (Bedrock, managed) | We do: weights on our disk, served by Ollama |
| How the code calls it | Converse API, authorized by IAM | Local HTTP, protected by network isolation |
| Agent code | `BedrockAdapter` | `OllamaAdapter`. **This is the only code difference** |
| Container images | ECR | A local registry, or `docker save` / `docker load` |
| Python packages, base images | Downloaded from PyPI and Docker Hub at build time | Must come from an internal mirror. There is no internet to download from |
| Identity | IAM roles with temporary credentials | Host accounts and file permissions. No IAM, no STS |
| Network control | VPC, security groups, endpoint policies | Host firewall, an internal-only Docker network |
| Logs | CloudWatch Logs | Local files (the week 8 `traces/*.jsonl`) or a self-hosted log stack |
| Model updates and end of life | AWS retires the model on its schedule; alerts warn us | The weights never disappear, but we must upgrade, patch, and re-test ourselves |
| Scaling | Add Fargate tasks; Bedrock provides model capacity | Buy hardware for peak load in advance |
| Alerts | SNS email | Internal mail or monitoring |

**What breaks in an air-gapped environment.** Anything that reaches the internet at runtime:

- **Package and image downloads.** `pip` / `uv` installs, Docker Hub pulls, and `ollama pull` all fail.
- **Managed services.** Bedrock, CloudWatch, IAM, and AWS Health alerts are gone.
- **Imported updates.** Every update (OS patches, base images, Python packages, model weights) has to come through a controlled import process, and vulnerability scanning has to run locally.

This design already avoids runtime downloads: dependencies are installed when the image is built, the model is pulled once, and the agent makes no external calls.

**Why a client might require it:**

- **Residency and sovereignty.** Applicant data (identity, driving history, insurance record) may not leave the country, or the premises.
- **Regulation or contracts.** A regulator, reinsurer, or contract may forbid sending this data to a third-party processor.
- **Lifecycle control.** The model cannot be retired out from under the client, as Gemma 3 27B will be on Bedrock on 30 March 2027.
- **Security posture.** With no internet path in or out, there is no exfiltration route.

## Comparison and recommendation

**Cost.**

- **AWS:** **$7.15 a day** (one-off tasks) or **$6.47 a day** (always-on service) at 10,000 requests. That is about $2,400–$2,600 a year, or roughly $0.0007 per request all-in. The model is about half of it. You pay only for use, and nothing up front.
- **On-prem:** $0 per token and no AWS bill. However:
  - The hardware must be sized for peak load, which means at least two machines like ours for 10,000 a day.
  - There is also power, space, maintenance, and staff time.
  - No hardware quote was made, so this is not priced here.

  At this volume the cloud is very likely cheaper. Owned hardware wins only at high, steady volume.

**Complexity.**

- **AWS:** more parts to configure: 2 roles, a VPC, 2 security groups, 5 endpoints, ECR, logs, ECS, and alerts. In return, AWS runs the model, the scaling, and the patching.
- **On-prem:** almost nothing new to configure (week 8 already runs). But we own everything: hardware, OS, Ollama and model updates, capacity, monitoring, and backups. Air-gapping adds an import process for every single update.

**Output quality.**

- **On-prem (measured):** all five cases landed in the expected tier with an acceptable decision. Output lengths were consistent, and week 8's prompts were tuned on this model.
- **AWS (not measured, because nothing was run):**
  - Gemma 3 27B has three times the parameters of qwen3.5:9b, so we expect it to follow the band rules at least as well.
  - However, the prompts were tuned on qwen, and Converse has no JSON mode, so more JSON repair calls are possible.
  - Either way, the code-enforced minimum score and tier ranges apply the same rules to both models.

  **How we would measure it:** run the same five fixtures through the Fargate task, then:

  1. Compare each score and tier with the table above.
  2. Check tiers against the manifest's expected tiers.
  3. Count repair calls (`llm_calls` above 1 in the log line).

| Client | Recommendation | Why |
|---|---|---|
| **Strict data residency** (data must stay in the country, or on premises) | **On-prem**, air-gapped if required. If in-country cloud is allowed, Bedrock **in-Region in that country** (ap-south-1 for India) with the same VPC-endpoint design | On-prem: no data leaves the boundary and the client controls the model's lifecycle. Bedrock in-Region: Gemma 3 27B has no cross-region routing, so requests stay in that Region |
| **No residency constraints** | **AWS Bedrock as an always-on service** | About $2,400 a year at 10,000 requests a day, little to operate, scales on demand, and AWS patches the infrastructure. The lifecycle alerts cover end of life |

## Cost at 10,000 requests a day

### Assumptions

| Item | Value | Source |
|---|---|---|
| Volume | 10,000 risk scoring requests per day. Month = 30 days, year = 365 days | Lab assumption |
| Tokens per request | 1,218 input, 204 output | Week 8 measured average for risk scoring (`docs/cost-and-latency.md`, traces 199–209 output tokens) |
| Gemma 3 27B on Bedrock | $0.23 per 1M input tokens, $0.38 per 1M output tokens | Given price |
| Fargate, Linux ARM64, us-east-1 | $0.03238 per vCPU-hour, $0.00356 per GB-hour | Public price lists |
| Task size | 0.25 vCPU, 0.5 GB | `aws/ecs/task-definition.json` |
| Interface VPC endpoint | $0.01 per endpoint per AZ-hour, $0.01 per GB processed | Public price lists |
| S3 gateway endpoint | Free | AWS |
| NAT gateway (for comparison) | $0.045 per hour, $0.045 per GB processed | Public price lists |
| CloudWatch Logs | $0.50 per GB ingested, $0.03 per GB-month stored, 30-day retention | Public price lists |
| ECR storage | $0.10 per GB-month, image about 0.1 GB compressed (estimate) | Public price lists |
| SQS (option B only) | $0.40 per million requests. The free tier is ignored | Public price lists |

One request costs **$0.000358** in Gemma tokens. One task-hour at 0.25 vCPU and 0.5 GB costs **$0.009875**.

### Two ways to run the agent

**Option A: one-off task per request.** Something upstream (the enrichment step) calls `ecs run-task` for every case. Fargate starts a container, the agent scores the case, prints the result, and the container stops. This is what `aws/deploy_steps.sh` does.

**Option B: always-on service.** ECS keeps 2 tasks running all the time, one in each Availability Zone. Enrichment puts each case on an SQS queue, and the running tasks take cases off the queue and score them. It needs:

- an SQS queue, and an SQS interface endpoint (a fifth endpoint)
- `sqs:ReceiveMessage` and `sqs:DeleteMessage` on that one queue for the task role
- a small change to the entrypoint, to read from the queue instead of file paths

| | A: one-off task | B: always-on service |
|---|---|---|
| Cost per day | **$7.15** | **$6.47** |
| Cost per month | **$214.47** | **$194.18** |
| Cost per year | **$2,609.41** | **$2,362.53** |
| Time per request | 30–60 s container start (image pull and boot), plus a few seconds of scoring | A few seconds of scoring. The container is already running |
| Containers started per day | 10,000 | 0 (2 always running, replaced only if one fails) |
| Fargate billing | 1-minute minimum per task × 10,000 | 2 tasks × 24 hours |
| Moving parts | Fewest: no queue | Queue, fifth endpoint, queue permissions |
| Capacity | Limited by how fast ECS can launch tasks | 2 tasks at about 5 s per case is about 34,000 cases a day, so 3× headroom |

**Option A line items**

| Item | Day | Month | Year |
|---|---:|---:|---:|
| Bedrock Gemma tokens | $3.58 | $107.30 | $1,305.46 |
| Fargate (1 min × 10,000 tasks) | $1.65 | $49.38 | $600.73 |
| VPC endpoints (4 × 2 AZ) | $1.92 | $57.60 | $700.80 |
| Endpoint data (about 0.1 GB a day) | $0.00 | $0.03 | $0.36 |
| CloudWatch Logs (about 10 MB a day) | $0.01 | $0.16 | $1.93 |
| ECR storage | $0.00 | $0.01 | $0.12 |
| **Total** | **$7.15** | **$214.47** | **$2,609.41** |

**Option B line items**

| Item | Day | Month | Year |
|---|---:|---:|---:|
| Bedrock Gemma tokens | $3.58 | $107.30 | $1,305.46 |
| Fargate (2 tasks × 24 h) | $0.47 | $14.22 | $173.01 |
| VPC endpoints (5 × 2 AZ) | $2.40 | $72.00 | $876.00 |
| Endpoint data | $0.00 | $0.03 | $0.36 |
| CloudWatch Logs | $0.01 | $0.16 | $1.93 |
| ECR storage | $0.00 | $0.01 | $0.12 |
| SQS (about 1.16M requests a month) | $0.02 | $0.46 | $5.64 |
| **Total** | **$6.47** | **$194.18** | **$2,362.53** |

### Decision

**For this lab we deploy option A, the one-off task.** The assignment asks for an ECS Fargate task. It needs no queue, so there are fewer permissions to grant and explain. It is enough to show that the agent runs and produces a result.

**At 10,000 requests a day we would choose option B, the always-on service.**

- It is about **$247 a year cheaper**.
- Each request takes seconds instead of waiting 30–60 s for a container to start.
- It avoids launching 10,000 containers a day.

The extra queue and endpoint are a small price for that. The scoring code does not change; only the entrypoint's input changes from files to queue messages.

### What drives the cost

- **The model is half the bill** ($3.58 of $7.15 a day in option A). Most of the 1,218 input tokens are the fixed system prompt with its three examples, not the case itself.
- **VPC endpoints are a fixed cost** of $1.92–$2.40 a day, whatever the volume. At low volume they would be most of the bill.
- **Endpoints are cheaper than NAT here, and private.** Two NAT gateways (one per AZ) cost $2.16 a day before data charges, against $1.92 for four endpoints, and NAT gives the task a path to the whole internet.
- **The S3 gateway endpoint matters a lot for option A.** Fargate does not cache images, so every one-off task pulls the full image. That is about 1,000 GB a day at 10,000 tasks. Through a NAT gateway this would add about $45 a day. Through the free S3 gateway endpoint it costs nothing.

### Ways to lower it

- **Shorter prompt.** Trimming the few-shot examples cuts input tokens on every request.
- **Bedrock Flex tier.** Gemma 3 27B supports Flex, a lower-cost tier for work that is not time-sensitive.
- **Batch inference.** If cases can be scored in batches instead of one at a time, Bedrock batch inference supports this model at a discounted price.
- **One AZ outside production.** Endpoints in one AZ instead of two halve the endpoint cost for test environments.
- **Fargate Spot for option B.** Cheaper tasks that can be interrupted. A case on the queue is simply picked up again.

### Not verified

- Prices come from public price lists, not a live AWS quote. Check them in the AWS Pricing Calculator.
- Token counts were measured with qwen3.5:9b. Gemma's tokenizer counts differently, so the real Gemma numbers will differ somewhat.
- The image size (about 0.1 GB) is an estimate. The image was not built.
- Option A assumes each task is billed the 1-minute minimum. Fargate bills from the start of the image pull to the end of the task, per second, with a 1-minute minimum.
- JSON repair calls (a second model call when the first reply is not valid JSON) are not included. Each one adds the cost of a request.

## Model lifecycle: end-of-life alerts

Gemma 3 27B (`google.gemma-3-27b-it`) is in the **Legacy** state on Bedrock, with an end-of-life date of **30 March 2027**. After that date the model can no longer be called, so every risk scoring request would fail. The team needs to hear about this well in advance, without relying on anyone remembering the date.

This section is a design only. Nothing was created in AWS.

### Design

```
 Reminders on dates we choose              Official notices from AWS
 EventBridge Scheduler (one-time)          EventBridge rule on AWS Health
   30 Nov 2026 09:00 New York  (4 months)    any Bedrock event for the account
   30 Dec 2026 09:00 New York  (3 months)
              │                                        │
              └──────────────┬─────────────────────────┘
                             ▼
              SNS topic  risk-agent-model-lifecycle
                             │
                             ▼
              email: underwriting-team@example.com
```

- **Two dated reminders.** The 4-month reminder (Monday 30 Nov 2026) is the heads-up to pick and test a replacement. The 3-month reminder (Wednesday 30 Dec 2026) is the deadline to have the switch planned. These fire on the dates even if AWS's own emails go to an inbox nobody reads.
- **AWS Health rule.** It catches anything a fixed date cannot: the end-of-life date moving earlier, or any other Bedrock notice for the account.
- **One SNS topic.** Both alerts send to the same topic, and the team subscribes once. The email address is a placeholder.
- **Outside the agent.** Alerting needs no change to the container and adds no permissions to the risk agent's task role.

EventBridge Scheduler was chosen over an EventBridge rule with a dated cron expression for three reasons:

- `at()` one-time schedules support a timezone. Cron rules run in UTC only.
- `--action-after-completion DELETE` removes each schedule after it fires, so nothing is left behind.
- Scheduler sends through its own IAM role, limited to publishing to this one topic.

### Least privilege

| Who | Can do | On |
|---|---|---|
| Scheduler role `risk-agent-lifecycle-scheduler` | `sns:Publish` | the `risk-agent-model-lifecycle` topic only |
| EventBridge rule `bedrock-health-notices` | `sns:Publish` (granted in the topic policy) | the same topic, and only when the request comes from this rule (`aws:SourceArn`) |

Scheduler role trust policy: only `scheduler.amazonaws.com`, and only for schedules in this account (`aws:SourceAccount`).

### Commands (simulated, us-east-1)

```bash
TOPIC_ARN="arn:aws:sns:us-east-1:ACCOUNT_ID:risk-agent-model-lifecycle"
ROLE_ARN="arn:aws:iam::ACCOUNT_ID:role/risk-agent-lifecycle-scheduler"
RULE_ARN="arn:aws:events:us-east-1:ACCOUNT_ID:rule/bedrock-health-notices"

# 1. Topic and email subscription. The recipient must click the confirmation link.
aws sns create-topic --name risk-agent-model-lifecycle \
  --tags Key=Project,Value=risk-scoring-agent
aws sns subscribe --topic-arn "$TOPIC_ARN" --protocol email \
  --notification-endpoint underwriting-team@example.com

# 2. Scheduler role: publish to this one topic, nothing else.
aws iam create-role --role-name risk-agent-lifecycle-scheduler \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "scheduler.amazonaws.com"},
      "Action": "sts:AssumeRole",
      "Condition": {"StringEquals": {"aws:SourceAccount": "ACCOUNT_ID"}}
    }]
  }'
aws iam put-role-policy --role-name risk-agent-lifecycle-scheduler \
  --policy-name publish-lifecycle-topic-only \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": "sns:Publish",
      "Resource": "arn:aws:sns:us-east-1:ACCOUNT_ID:risk-agent-model-lifecycle"
    }]
  }'

# 3. Two one-time reminders. Each deletes itself after it fires.
MESSAGE='Gemma 3 27B (google.gemma-3-27b-it) on Amazon Bedrock reaches end of life on 30 March 2027. After that date the risk scoring agent cannot score cases. To do: (1) choose a replacement model, for example Gemma 4 on Bedrock; (2) score the five week9/fixtures with it and compare with the current results; (3) update BEDROCK_MODEL_ID in the task definition, the model ARN in the task role policy, and the Bedrock endpoint policy; (4) push a new image tag and run the task.'

for reminder in "4-months 2026-11-30T09:00:00" "3-months 2026-12-30T09:00:00"; do
  name="${reminder%% *}"
  when="${reminder##* }"
  aws scheduler create-schedule --name "gemma3-eol-reminder-${name}" \
    --schedule-expression "at(${when})" \
    --schedule-expression-timezone America/New_York \
    --flexible-time-window Mode=OFF \
    --action-after-completion DELETE \
    --target "$(jq -n --arg arn "$TOPIC_ARN" --arg role "$ROLE_ARN" --arg msg "$MESSAGE" \
      '{Arn: $arn, RoleArn: $role, Input: $msg}')"
done

# 4. AWS Health rule: any Bedrock Health event goes to the topic.
aws events put-rule --name bedrock-health-notices \
  --event-pattern '{
    "source": ["aws.health"],
    "detail-type": ["AWS Health Event"],
    "detail": {"service": ["BEDROCK"]}
  }' \
  --tags Key=Project,Value=risk-scoring-agent
aws events put-targets --rule bedrock-health-notices \
  --targets "Id=lifecycle-topic,Arn=${TOPIC_ARN}"

# Topic policy: only this rule may publish through EventBridge.
aws sns set-topic-attributes --topic-arn "$TOPIC_ARN" --attribute-name Policy \
  --attribute-value '{
    "Version": "2012-10-17",
    "Statement": [{
      "Sid": "OnlyTheBedrockHealthRule",
      "Effect": "Allow",
      "Principal": {"Service": "events.amazonaws.com"},
      "Action": "sns:Publish",
      "Resource": "arn:aws:sns:us-east-1:ACCOUNT_ID:risk-agent-model-lifecycle",
      "Condition": {"ArnEquals": {"aws:SourceArn": "arn:aws:events:us-east-1:ACCOUNT_ID:rule/bedrock-health-notices"}}
    }]
  }'
```

### Checking the status by hand

At any time, this shows whether the model is `ACTIVE` or `LEGACY`:

```bash
aws bedrock get-foundation-model --model-identifier google.gemma-3-27b-it \
  --query 'modelDetails.modelLifecycle.status' --output text
```

### Teardown

```bash
aws scheduler delete-schedule --name gemma3-eol-reminder-4-months   # only if it has not fired yet
aws scheduler delete-schedule --name gemma3-eol-reminder-3-months   # only if it has not fired yet
aws events remove-targets --rule bedrock-health-notices --ids lifecycle-topic
aws events delete-rule --name bedrock-health-notices
aws iam delete-role-policy --role-name risk-agent-lifecycle-scheduler --policy-name publish-lifecycle-topic-only
aws iam delete-role --role-name risk-agent-lifecycle-scheduler
aws sns delete-topic --topic-arn "$TOPIC_ARN"
```

### Cost

Close to zero: two scheduled invocations, one EventBridge rule, and a few emails. No compute runs.

### Not verified

- The AWS Health service code for Bedrock is assumed to be `BEDROCK`. If it is different, the rule never matches and fails silently. Before relying on the rule, confirm the code from the AWS Health sample events in the EventBridge console.
- AWS Health delivers events to EventBridge in the Region they apply to. The rule is in us-east-1, the Region the agent runs in.

## What was not verified

Nothing was run in AWS. **Checked:**

- All 16 week 9 tests pass, with the week 8 suite, in CI.
- Every policy and task definition file is valid JSON.
- The deploy script's syntax is valid, and its guard stops it from running.
- The image's dependency install works, outside Docker.
- A real boto3 Bedrock client is created with the intended settings: us-east-1, 30 s read timeout, one attempt.
- The fixtures match the traced enrichment output.

**Not confirmed:**

- **AWS accepts the commands.** Only a real account can confirm that every command and policy is accepted as written.
- **Docker build.** `docker build` was never run, so the ignore patterns, `COPY` paths, and user creation are untested.
- **Gemma's behaviour.** Whether Bedrock accepts a separate system prompt for Gemma (the `BEDROCK_SYSTEM_IN_USER` switch covers it either way), and Gemma's real output quality, token counts, latency, and JSON reliability.
- **Prices.** They come from public price lists, not a live AWS quote.
- **The AWS Health service code** `BEDROCK` used in the lifecycle rule.
- **The image size** (about 0.1 GB), which is estimated.

## Files

| Path | What it is |
|---|---|
| [`build_fixtures.py`](build_fixtures.py) | Builds the five `EnrichedCase` inputs from the week 8 agents and checks them against `traces/` |
| [`fixtures/`](fixtures/) | The five inputs both deployments score |
| [`risk_agent/bedrock_adapter.py`](risk_agent/bedrock_adapter.py) | `LLMAdapter` for Bedrock's Converse API |
| [`risk_agent/run_risk_agent.py`](risk_agent/run_risk_agent.py) | Container entrypoint: scores files, prints one JSON line per case |
| [`tests/`](tests/) | 16 tests with fake clients |
| [`Dockerfile`](Dockerfile), [`Dockerfile.dockerignore`](Dockerfile.dockerignore) | The image and its build-context allow-list |
| [`aws/iam/`](aws/iam/) | Trust and permission policies for the execution and task roles |
| [`aws/network/`](aws/network/) | Bedrock and S3 endpoint policies |
| [`aws/ecs/task-definition.json`](aws/ecs/task-definition.json) | Fargate task definition |
| [`aws/deploy_steps.sh`](aws/deploy_steps.sh) | Every deployment command in order, plus teardown. Never run |
