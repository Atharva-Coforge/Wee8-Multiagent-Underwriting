#!/usr/bin/env bash
# Simulated AWS deployment, us-east-1: network, IAM, ECR, logs, ECS Fargate.
# Documentation only. Never run: the lab does not allow cloud services.
#
# Order: the network comes first, because the IAM policies name the VPC and
# the Bedrock endpoint it creates. IAM still comes before any compute: no task
# runs until both roles exist and have been checked.
#
#   CONFIRM_REAL_AWS=yes bash week9/aws/deploy_steps.sh            # deploy and run once
#   CONFIRM_REAL_AWS=yes bash week9/aws/deploy_steps.sh teardown   # delete everything
#
# The identity running this script is a deployer with IAM, EC2, ECR, ECS, and
# CloudWatch Logs permissions. It is not the task role, which can only call
# Gemma.

set -euo pipefail

if [[ "${CONFIRM_REAL_AWS:-}" != "yes" ]]; then
  echo "Simulation only: this script creates billable AWS resources." >&2
  echo "Set CONFIRM_REAL_AWS=yes to run it against a real account." >&2
  exit 1
fi

REGION="us-east-1"
PROJECT="risk-scoring-agent"
REPO="risk-scoring-agent"
TAG="v1"
CLUSTER="risk-scoring"
LOG_GROUP="/ecs/risk-scoring-agent"
EXEC_ROLE="risk-agent-execution-role"
TASK_ROLE="risk-agent-task-role"
EXEC_POLICY="risk-agent-execution-policy"
TASK_POLICY="risk-agent-task-policy"
MODEL_ARN="arn:aws:bedrock:${REGION}::foundation-model/google.gemma-3-27b-it"

export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AWS_DIR="$ROOT/week9/aws"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}"
TAGS="Key=Project,Value=${PROJECT}"

# Set by the network step, then written into the IAM policies.
VPC_ID=""
BEDROCK_VPCE_ID=""

# The JSON files use the placeholders ACCOUNT_ID, VPC_ID, and BEDROCK_VPCE_ID.
# Fill in the real values into a temp copy, and stop if any are still missing.
RENDERED="$(mktemp -d)"
trap 'rm -rf "$RENDERED"' EXIT
render() {
  local out
  out="$RENDERED/$(basename "$1")"
  sed -e "s/ACCOUNT_ID/${ACCOUNT_ID}/g" \
    -e "s/BEDROCK_VPCE_ID/${BEDROCK_VPCE_ID}/g" \
    -e "s/VPC_ID/${VPC_ID}/g" \
    "$AWS_DIR/$1" > "$out"
  if grep -qE '""|ACCOUNT_ID|BEDROCK_VPCE_ID|VPC_ID' "$out"; then
    echo "render: $1 still has an empty or unfilled placeholder" >&2
    exit 1
  fi
  echo "$out"
}

deploy() {
  # ---------------------------------------------------------------------------
  # 1. Network: a VPC with two private subnets, no internet gateway, no NAT.
  #    AWS services are reached through VPC endpoints only.
  # ---------------------------------------------------------------------------
  # Pick AZs where the Bedrock runtime endpoint is offered.
  aws ec2 describe-vpc-endpoint-services \
    --service-names "com.amazonaws.${REGION}.bedrock-runtime" \
    --query 'ServiceDetails[0].AvailabilityZones' --output text

  local subnet_a subnet_b rtb_id task_sg endpoint_sg s3_prefix_list
  VPC_ID="$(aws ec2 create-vpc --cidr-block 10.0.0.0/16 \
    --tag-specifications "ResourceType=vpc,Tags=[{$TAGS},{Key=Name,Value=risk-agent-vpc}]" \
    --query Vpc.VpcId --output text)"
  # Private DNS for interface endpoints needs both of these.
  aws ec2 modify-vpc-attribute --vpc-id "$VPC_ID" --enable-dns-support '{"Value":true}'
  aws ec2 modify-vpc-attribute --vpc-id "$VPC_ID" --enable-dns-hostnames '{"Value":true}'

  subnet_a="$(aws ec2 create-subnet --vpc-id "$VPC_ID" --cidr-block 10.0.1.0/24 \
    --availability-zone "${REGION}a" \
    --tag-specifications "ResourceType=subnet,Tags=[{$TAGS},{Key=Name,Value=risk-agent-private-a}]" \
    --query Subnet.SubnetId --output text)"
  subnet_b="$(aws ec2 create-subnet --vpc-id "$VPC_ID" --cidr-block 10.0.2.0/24 \
    --availability-zone "${REGION}b" \
    --tag-specifications "ResourceType=subnet,Tags=[{$TAGS},{Key=Name,Value=risk-agent-private-b}]" \
    --query Subnet.SubnetId --output text)"

  # Route table with only the local route: nothing can leave the VPC.
  rtb_id="$(aws ec2 create-route-table --vpc-id "$VPC_ID" \
    --tag-specifications "ResourceType=route-table,Tags=[{$TAGS},{Key=Name,Value=risk-agent-private}]" \
    --query RouteTable.RouteTableId --output text)"
  aws ec2 associate-route-table --route-table-id "$rtb_id" --subnet-id "$subnet_a"
  aws ec2 associate-route-table --route-table-id "$rtb_id" --subnet-id "$subnet_b"

  # Security groups. Both start with no inbound rules; remove the default
  # allow-all outbound rule, then add only what is needed.
  task_sg="$(aws ec2 create-security-group --vpc-id "$VPC_ID" \
    --group-name risk-agent-task-sg \
    --description "Risk agent task: no inbound, HTTPS out to endpoints only" \
    --tag-specifications "ResourceType=security-group,Tags=[{$TAGS}]" \
    --query GroupId --output text)"
  endpoint_sg="$(aws ec2 create-security-group --vpc-id "$VPC_ID" \
    --group-name risk-agent-endpoint-sg \
    --description "VPC endpoints: HTTPS in from the risk agent task only" \
    --tag-specifications "ResourceType=security-group,Tags=[{$TAGS}]" \
    --query GroupId --output text)"
  for sg in "$task_sg" "$endpoint_sg"; do
    aws ec2 revoke-security-group-egress --group-id "$sg" \
      --ip-permissions '[{"IpProtocol":"-1","IpRanges":[{"CidrIp":"0.0.0.0/0"}]}]'
  done

  # Endpoints accept HTTPS only from the task security group.
  aws ec2 authorize-security-group-ingress --group-id "$endpoint_sg" \
    --ip-permissions "[{\"IpProtocol\":\"tcp\",\"FromPort\":443,\"ToPort\":443,\"UserIdGroupPairs\":[{\"GroupId\":\"${task_sg}\"}]}]"
  # Task sends HTTPS only to the endpoints and to S3 (image layers).
  aws ec2 authorize-security-group-egress --group-id "$task_sg" \
    --ip-permissions "[{\"IpProtocol\":\"tcp\",\"FromPort\":443,\"ToPort\":443,\"UserIdGroupPairs\":[{\"GroupId\":\"${endpoint_sg}\"}]}]"
  s3_prefix_list="$(aws ec2 describe-managed-prefix-lists \
    --filters "Name=prefix-list-name,Values=com.amazonaws.${REGION}.s3" \
    --query 'PrefixLists[0].PrefixListId' --output text)"
  aws ec2 authorize-security-group-egress --group-id "$task_sg" \
    --ip-permissions "[{\"IpProtocol\":\"tcp\",\"FromPort\":443,\"ToPort\":443,\"PrefixListIds\":[{\"PrefixListId\":\"${s3_prefix_list}\"}]}]"

  # Interface endpoints: image pull (ecr.api, ecr.dkr), logs, and Bedrock.
  for service in ecr.api ecr.dkr logs; do
    aws ec2 create-vpc-endpoint --vpc-id "$VPC_ID" --vpc-endpoint-type Interface \
      --service-name "com.amazonaws.${REGION}.${service}" \
      --subnet-ids "$subnet_a" "$subnet_b" --security-group-ids "$endpoint_sg" \
      --private-dns-enabled \
      --tag-specifications "ResourceType=vpc-endpoint,Tags=[{$TAGS}]"
  done
  # The endpoint policy checks the caller with aws:PrincipalArn, so it can be
  # set now, before the task role exists.
  BEDROCK_VPCE_ID="$(aws ec2 create-vpc-endpoint --vpc-id "$VPC_ID" --vpc-endpoint-type Interface \
    --service-name "com.amazonaws.${REGION}.bedrock-runtime" \
    --subnet-ids "$subnet_a" "$subnet_b" --security-group-ids "$endpoint_sg" \
    --private-dns-enabled \
    --policy-document "file://$(render network/bedrock-endpoint-policy.json)" \
    --tag-specifications "ResourceType=vpc-endpoint,Tags=[{$TAGS}]" \
    --query VpcEndpoint.VpcEndpointId --output text)"
  # Gateway endpoint for S3, where ECR stores image layers. Free.
  aws ec2 create-vpc-endpoint --vpc-id "$VPC_ID" --vpc-endpoint-type Gateway \
    --service-name "com.amazonaws.${REGION}.s3" --route-table-ids "$rtb_id" \
    --policy-document "file://$AWS_DIR/network/s3-gateway-endpoint-policy.json" \
    --tag-specifications "ResourceType=vpc-endpoint,Tags=[{$TAGS}]"

  # ---------------------------------------------------------------------------
  # 2. IAM: two roles, each with one customer managed policy. The policies are
  #    tied to the VPC and the Bedrock endpoint from step 1.
  # ---------------------------------------------------------------------------
  # Execution role: used by ECS/Fargate to pull the image and write logs.
  aws iam create-role --role-name "$EXEC_ROLE" \
    --assume-role-policy-document "file://$(render iam/execution-role-trust-policy.json)" \
    --description "ECS pulls the risk agent image and writes its logs" \
    --tags "$TAGS"
  aws iam create-policy --policy-name "$EXEC_POLICY" \
    --policy-document "file://$(render iam/execution-role-policy.json)" \
    --tags "$TAGS"
  aws iam attach-role-policy --role-name "$EXEC_ROLE" \
    --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${EXEC_POLICY}"

  # Task role: used by our code inside the container. Gemma only, and only
  # through our Bedrock endpoint.
  aws iam create-role --role-name "$TASK_ROLE" \
    --assume-role-policy-document "file://$(render iam/task-role-trust-policy.json)" \
    --description "Risk scoring agent: invoke Gemma 3 27B in us-east-1 through our endpoint only" \
    --tags "$TAGS"
  aws iam create-policy --policy-name "$TASK_POLICY" \
    --policy-document "file://$(render iam/task-role-policy.json)" \
    --tags "$TAGS"
  aws iam attach-role-policy --role-name "$TASK_ROLE" \
    --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${TASK_POLICY}"

  # Confirm the roles exist and have the right policy before anything uses them.
  for role in "$EXEC_ROLE" "$TASK_ROLE"; do
    aws iam get-role --role-name "$role" --query 'Role.Arn' --output text
    aws iam list-attached-role-policies --role-name "$role" \
      --query 'AttachedPolicies[].PolicyName' --output text
  done
  aws iam get-policy-version \
    --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${TASK_POLICY}" \
    --version-id v1 --query 'PolicyVersion.Document'

  # Least privilege check on the task role. Expected, in order:
  # allowed, implicitDeny, implicitDeny, implicitDeny.
  local task_role_arn="arn:aws:iam::${ACCOUNT_ID}:role/${TASK_ROLE}"
  local through_endpoint="ContextKeyName=aws:SourceVpce,ContextKeyValues=${BEDROCK_VPCE_ID},ContextKeyType=string"
  # Gemma 3 27B, called through our endpoint.
  aws iam simulate-principal-policy --policy-source-arn "$task_role_arn" \
    --action-names bedrock:InvokeModel --resource-arns "$MODEL_ARN" \
    --context-entries "$through_endpoint" \
    --query 'EvaluationResults[].[EvalActionName,EvalDecision]' --output text
  # Gemma 3 27B, called from anywhere else (no endpoint in the request).
  aws iam simulate-principal-policy --policy-source-arn "$task_role_arn" \
    --action-names bedrock:InvokeModel --resource-arns "$MODEL_ARN" \
    --query 'EvaluationResults[].[EvalActionName,EvalDecision]' --output text
  # A different model, even through our endpoint.
  aws iam simulate-principal-policy --policy-source-arn "$task_role_arn" \
    --action-names bedrock:InvokeModel \
    --resource-arns "arn:aws:bedrock:${REGION}::foundation-model/google.gemma-3-12b-it" \
    --context-entries "$through_endpoint" \
    --query 'EvaluationResults[].[EvalActionName,EvalDecision]' --output text
  # Any other service.
  aws iam simulate-principal-policy --policy-source-arn "$task_role_arn" \
    --action-names s3:GetObject --resource-arns "arn:aws:s3:::any-bucket/any-key" \
    --query 'EvaluationResults[].[EvalActionName,EvalDecision]' --output text

  # ---------------------------------------------------------------------------
  # 3. ECR: private repository, immutable tags, scan on push. Build and push.
  # ---------------------------------------------------------------------------
  aws ecr create-repository --repository-name "$REPO" \
    --image-tag-mutability IMMUTABLE \
    --image-scanning-configuration scanOnPush=true \
    --tags "$TAGS"
  aws ecr get-login-password \
    | docker login --username AWS --password-stdin \
      "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
  docker build --platform linux/arm64 -f "$ROOT/week9/Dockerfile" -t "$IMAGE_URI" "$ROOT"
  docker push "$IMAGE_URI"
  aws ecr wait image-scan-complete --repository-name "$REPO" --image-id "imageTag=${TAG}"
  aws ecr describe-image-scan-findings --repository-name "$REPO" \
    --image-id "imageTag=${TAG}" --query 'imageScanFindings.findingSeverityCounts'

  # ---------------------------------------------------------------------------
  # 4. CloudWatch Logs: created up front, kept for 30 days.
  # ---------------------------------------------------------------------------
  aws logs create-log-group --log-group-name "$LOG_GROUP" --tags "Project=${PROJECT}"
  aws logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 30

  # ---------------------------------------------------------------------------
  # 5. ECS: cluster, task definition with both roles set, one Fargate run.
  # ---------------------------------------------------------------------------
  aws ecs create-cluster --cluster-name "$CLUSTER" --tags "key=Project,value=${PROJECT}"
  aws ecs register-task-definition \
    --cli-input-json "file://$(render ecs/task-definition.json)"

  local task_arn
  task_arn="$(aws ecs run-task --cluster "$CLUSTER" --launch-type FARGATE \
    --platform-version LATEST --task-definition "$PROJECT" --count 1 \
    --network-configuration "awsvpcConfiguration={subnets=[${subnet_a},${subnet_b}],securityGroups=[${task_sg}],assignPublicIp=DISABLED}" \
    --query 'tasks[0].taskArn' --output text)"
  # To score a single case instead of all five, add:
  #   --overrides '{"containerOverrides":[{"name":"risk-scoring-agent","command":["week9/fixtures/tyler_brandt.json"]}]}'

  # ---------------------------------------------------------------------------
  # 6. Confirm: exit code 0 means every case was scored. Results are in the logs.
  # ---------------------------------------------------------------------------
  aws ecs wait tasks-stopped --cluster "$CLUSTER" --tasks "$task_arn"
  aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$task_arn" \
    --query 'tasks[0].{status:lastStatus,stopCode:stopCode,reason:stoppedReason,exitCode:containers[0].exitCode}'
  aws logs tail "$LOG_GROUP" --since 30m
}

teardown() {
  # Reverse order. Resources are found by their Project tag or fixed names.
  local vpc_id
  vpc_id="$(aws ec2 describe-vpcs --filters "Name=tag:Project,Values=${PROJECT}" \
    --query 'Vpcs[0].VpcId' --output text)"

  for arn in $(aws ecs list-task-definitions --family-prefix "$PROJECT" \
    --query 'taskDefinitionArns[]' --output text); do
    aws ecs deregister-task-definition --task-definition "$arn" > /dev/null
  done
  aws ecs delete-cluster --cluster "$CLUSTER"
  aws logs delete-log-group --log-group-name "$LOG_GROUP"

  if [[ "$vpc_id" != "None" ]]; then
    # Interface endpoints are billed per hour, so they go first.
    local endpoints
    endpoints="$(aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=${vpc_id}" \
      --query 'VpcEndpoints[].VpcEndpointId' --output text)"
    if [[ -n "$endpoints" ]]; then
      # shellcheck disable=SC2086  # one argument per endpoint id
      aws ec2 delete-vpc-endpoints --vpc-endpoint-ids $endpoints
      while [[ -n "$(aws ec2 describe-vpc-endpoints \
        --filters "Name=vpc-id,Values=${vpc_id}" "Name=vpc-endpoint-state,Values=deleting" \
        --query 'VpcEndpoints[].VpcEndpointId' --output text)" ]]; do
        sleep 10
      done
    fi
    # The two groups reference each other, so revoke their rules before deleting.
    local groups
    groups="$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=${vpc_id}" \
      "Name=tag:Project,Values=${PROJECT}" --query 'SecurityGroups[].GroupId' --output text)"
    for sg in $groups; do
      local ingress egress
      ingress="$(aws ec2 describe-security-group-rules --filters "Name=group-id,Values=${sg}" \
        --query 'SecurityGroupRules[?!IsEgress].SecurityGroupRuleId' --output text)"
      egress="$(aws ec2 describe-security-group-rules --filters "Name=group-id,Values=${sg}" \
        --query 'SecurityGroupRules[?IsEgress].SecurityGroupRuleId' --output text)"
      # shellcheck disable=SC2086  # one argument per rule id
      if [[ -n "$ingress" ]]; then
        aws ec2 revoke-security-group-ingress --group-id "$sg" --security-group-rule-ids $ingress
      fi
      # shellcheck disable=SC2086
      if [[ -n "$egress" ]]; then
        aws ec2 revoke-security-group-egress --group-id "$sg" --security-group-rule-ids $egress
      fi
    done
    for sg in $groups; do
      aws ec2 delete-security-group --group-id "$sg"
    done
    for subnet in $(aws ec2 describe-subnets --filters "Name=vpc-id,Values=${vpc_id}" \
      --query 'Subnets[].SubnetId' --output text); do
      aws ec2 delete-subnet --subnet-id "$subnet"
    done
    for rtb in $(aws ec2 describe-route-tables --filters "Name=vpc-id,Values=${vpc_id}" \
      "Name=tag:Project,Values=${PROJECT}" --query 'RouteTables[].RouteTableId' --output text); do
      aws ec2 delete-route-table --route-table-id "$rtb"
    done
    aws ec2 delete-vpc --vpc-id "$vpc_id"
  fi

  aws ecr delete-repository --repository-name "$REPO" --force

  for pair in "${EXEC_ROLE}:${EXEC_POLICY}" "${TASK_ROLE}:${TASK_POLICY}"; do
    local role="${pair%%:*}" policy_arn="arn:aws:iam::${ACCOUNT_ID}:policy/${pair##*:}"
    aws iam detach-role-policy --role-name "$role" --policy-arn "$policy_arn"
    aws iam delete-policy --policy-arn "$policy_arn"
    aws iam delete-role --role-name "$role"
  done
}

case "${1:-deploy}" in
  deploy) deploy ;;
  teardown) teardown ;;
  *) echo "usage: $0 [deploy|teardown]" >&2; exit 2 ;;
esac
