# Week 9: Deploying the Risk Scoring Agent

Deployment write-up for the risk scoring agent: simulated AWS (Bedrock Gemma 3 27B on ECS Fargate, us-east-1) and simulated on-prem (Ollama qwen3.5:9b).

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
