#!/bin/bash
# =============================================================================
# Create SageMaker Execution Role for MedML Training Dashboard
# =============================================================================
# Bash equivalent of the shared Python get_or_create_role() utility.
# Creates (or updates) the IAM role with managed policies and an inline
# MLflow policy, matching the behavior of the Python helper.
#
# Usage:
#   ./create-sagemaker-role.sh [role-name]
#
# Examples:
#   ./create-sagemaker-role.sh
#   ./create-sagemaker-role.sh AmazonSageMaker-ExecutionRole-sgm
#
# Prerequisites:
#   - AWS CLI configured with iam:CreateRole, iam:AttachRolePolicy,
#     iam:PutRolePolicy permissions
#   - .env.local must exist with AWS_ACCOUNT_ID set (for .env.local update)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_LOCAL="$SCRIPT_DIR/.env.local"

# --- Configuration -----------------------------------------------------------
ROLE_NAME="${1:-AmazonSageMaker-ExecutionRole-sgm}"

# Managed policies to attach (matches Python get_or_create_role defaults)
MANAGED_POLICIES=(
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
    "arn:aws:iam::aws:policy/AmazonS3FullAccess"
    "arn:aws:iam::aws:policy/AmazonRekognitionFullAccess"
    "arn:aws:iam::aws:policy/ComprehendFullAccess"
    "arn:aws:iam::aws:policy/ComprehendMedicalFullAccess"
    "arn:aws:iam::aws:policy/AmazonElasticContainerRegistryPublicFullAccess"
    "arn:aws:iam::aws:policy/AmazonSageMakerCanvasFullAccess"
    "arn:aws:iam::aws:policy/AmazonSageMakerCanvasAIServicesAccess"
    "arn:aws:iam::aws:policy/AmazonSageMakerCanvasSMDataScienceAssistantAccess"
    "arn:aws:iam::aws:policy/AmazonSageMakerCanvasDataPrepFullAccess"
)

# --- Load .env.local (optional, for updating SAGEMAKER_ROLE_ARN) -------------
AWS_ACCOUNT_ID=""
AWS_DEFAULT_REGION=""
MLFLOW_TRACKING_SERVER_NAME=""
if [[ -f "$ENV_LOCAL" ]]; then
    eval "$(grep -E '^(AWS_ACCOUNT_ID|AWS_DEFAULT_REGION|MLFLOW_TRACKING_SERVER_NAME)=' "$ENV_LOCAL" 2>/dev/null)" || true
fi

REGION="${AWS_DEFAULT_REGION:-us-east-1}"

echo "============================================================"
echo "  MedML SageMaker Execution Role Setup"
echo "============================================================"
echo "  Role Name:    $ROLE_NAME"
echo "  Region:       $REGION"
echo "  Policies:     ${#MANAGED_POLICIES[@]} managed + 1 inline (MLflow)"
if [[ -n "$MLFLOW_TRACKING_SERVER_NAME" ]]; then
echo "  MLflow scope: $MLFLOW_TRACKING_SERVER_NAME"
else
echo "  MLflow scope: all tracking servers (set MLFLOW_TRACKING_SERVER_NAME to restrict)"
fi
echo "============================================================"
echo ""

# --- Trust policy -------------------------------------------------------------
TRUST_POLICY=$(mktemp)
MLFLOW_POLICY=$(mktemp)
trap 'rm -f "$TRUST_POLICY" "$MLFLOW_POLICY"' EXIT

cat > "$TRUST_POLICY" <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# --- Inline MLflow policy -----------------------------------------------------
# Replace Resource with your tracking server ARN for least-privilege access.
# If MLFLOW_TRACKING_SERVER_NAME is set in .env.local, the policy is scoped
# to that server. Otherwise, it scopes to all tracking servers in the account.
if [[ -n "$AWS_ACCOUNT_ID" && -n "$MLFLOW_TRACKING_SERVER_NAME" ]]; then
    MLFLOW_RESOURCE="arn:aws:sagemaker:${REGION}:${AWS_ACCOUNT_ID}:mlflow-tracking-server/${MLFLOW_TRACKING_SERVER_NAME}"
elif [[ -n "$AWS_ACCOUNT_ID" ]]; then
    MLFLOW_RESOURCE="arn:aws:sagemaker:${REGION}:${AWS_ACCOUNT_ID}:mlflow-tracking-server/*"
else
    # Fallback — will be replaced when AWS_ACCOUNT_ID is available
    MLFLOW_RESOURCE="arn:aws:sagemaker:${REGION}:*:mlflow-tracking-server/*"
fi

cat > "$MLFLOW_POLICY" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "MLflowTrackingAccess",
      "Effect": "Allow",
      "Action": [
        "sagemaker-mlflow:*"
      ],
      "Resource": "${MLFLOW_RESOURCE}"
    }
  ]
}
EOF

# --- Create or update the IAM role -------------------------------------------
echo "📋 Step 1: Checking if role '$ROLE_NAME' exists..."

if aws iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
    ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)
    echo "   ⚠️  Role already exists: $ROLE_ARN"
    echo "   Updating trust policy..."
    aws iam update-assume-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-document "file://$TRUST_POLICY"
    echo "   ✅ Trust policy updated."
else
    echo "   Creating role..."
    ROLE_ARN=$(aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document "file://$TRUST_POLICY" \
        --description "SageMaker execution role for medical imaging training" \
        --query 'Role.Arn' --output text)
    echo "   ✅ Created role: $ROLE_ARN"
fi

# --- Attach managed policies --------------------------------------------------
echo ""
echo "📋 Step 2: Attaching managed policies..."

for POLICY_ARN in "${MANAGED_POLICIES[@]}"; do
    POLICY_SHORT="${POLICY_ARN##*/}"
    aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn "$POLICY_ARN"
    echo "   ✅ Attached: $POLICY_SHORT"
done

# --- Attach inline MLflow policy ----------------------------------------------
echo ""
echo "📋 Step 3: Adding inline MLflow policy..."

aws iam put-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name "mlflow-policy" \
    --policy-document "file://$MLFLOW_POLICY"

echo "   ✅ Added inline policy: mlflow-policy"

# --- Update .env.local with the role ARN -------------------------------------
if [[ -f "$ENV_LOCAL" ]]; then
    echo ""
    echo "📋 Step 4: Updating .env.local with role ARN..."

    if grep -q "^SAGEMAKER_ROLE_ARN=" "$ENV_LOCAL"; then
        sed -i.bak "s|^SAGEMAKER_ROLE_ARN=.*|SAGEMAKER_ROLE_ARN=${ROLE_ARN}|" "$ENV_LOCAL"
        rm -f "${ENV_LOCAL}.bak"
        echo "   ✅ Updated SAGEMAKER_ROLE_ARN in .env.local"
    else
        echo "" >> "$ENV_LOCAL"
        echo "SAGEMAKER_ROLE_ARN=${ROLE_ARN}" >> "$ENV_LOCAL"
        echo "   ✅ Added SAGEMAKER_ROLE_ARN to .env.local"
    fi
fi

# --- Done --------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  ✅ Done!"
echo "============================================================"
echo ""
echo "  Role ARN: $ROLE_ARN"
echo ""
echo "  Managed policies attached:"
for POLICY_ARN in "${MANAGED_POLICIES[@]}"; do
    echo "    - ${POLICY_ARN##*/}"
done
echo "  Inline policies:"
echo "    - mlflow-policy"
echo ""
echo "============================================================"
