#!/bin/bash
# =============================================================================
# Integration Test Pipeline
# =============================================================================
# End-to-end test of the MedML Training Dashboard deployment pipeline.
# Tests both classification and segmentation endpoints.
#
# Usage:
#   ./tests/integration/run_pipeline_test.sh                    # Run all tests
#   ./tests/integration/run_pipeline_test.sh --segmentation     # Segmentation only
#   ./tests/integration/run_pipeline_test.sh --classification   # Classification only
#   ./tests/integration/run_pipeline_test.sh --skip-deploy      # Test existing endpoints
#
# Prerequisites:
#   - .env.local configured with AWS_ACCOUNT_ID, SAGEMAKER_ROLE_ARN
#   - Docker images built and pushed (build_and_push.sh, build_and_push_inference.sh)
#   - Training data available in S3
#   - Test images available in S3
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# Load config
if [[ -f .env.local ]]; then
    eval "$(grep -E '^(AWS_ACCOUNT_ID|AWS_DEFAULT_REGION|SAGEMAKER_ROLE_ARN|TEST_IMAGE_S3_URI|TEST_SEGMENTATION_S3_URI)=' .env.local)"
fi

# ── Configuration ─────────────────────────────────────────────────────────────
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
ROLE_ARN="${SAGEMAKER_ROLE_ARN:-arn:aws:iam::${AWS_ACCOUNT_ID}:role/SageMakerExecutionRole}"

# Segmentation
SEG_ENDPOINT="segmentation-integration-test"
SEG_TEST_IMAGE="${TEST_SEGMENTATION_S3_URI:-s3://test-imaging-workflows/segmentation/test/10090344/img.nii.gz}"
SEG_MODEL_S3="${SEG_MODEL_S3:-}"  # Set via env or auto-detect latest

# Classification
CLASS_ENDPOINT="classification-integration-test"
CLASS_TEST_IMAGE="${TEST_IMAGE_S3_URI:-s3://test-imaging-workflows/classification/vindr-spinexr-subset/test/No_finding/1a3de4a48b92b4f8f5ea.dicom}"
CLASS_MODEL_S3="${CLASS_MODEL_S3:-}"  # Set via env or auto-detect latest

# Flags
RUN_SEG=true
RUN_CLASS=true
SKIP_DEPLOY=false
CLEANUP=true

for arg in "$@"; do
    case "$arg" in
        --segmentation)   RUN_CLASS=false ;;
        --classification) RUN_SEG=false ;;
        --skip-deploy)    SKIP_DEPLOY=true ;;
        --no-cleanup)     CLEANUP=false ;;
        --help|-h)
            echo "Usage: $0 [--segmentation|--classification|--skip-deploy|--no-cleanup]"
            exit 0 ;;
    esac
done

# ── Utilities ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
PASS=0
FAIL=0

pass() { echo -e "${GREEN}✅ PASS:${NC} $1"; ((PASS++)); }
fail() { echo -e "${RED}❌ FAIL:${NC} $1"; ((FAIL++)); }
info() { echo -e "${YELLOW}ℹ️  $1${NC}"; }

# ── Auto-detect latest model artifacts ────────────────────────────────────────
if [[ -z "$SEG_MODEL_S3" ]]; then
    SEG_MODEL_S3=$(aws s3 ls s3://medml-dashboard/model-output/segmentation/ --region "$REGION" | grep "segmentation-segresnet" | sort | tail -1 | awk '{print "s3://medml-dashboard/model-output/segmentation/" $NF "output/model.tar.gz"}' 2>/dev/null || echo "")
    if [[ -n "$SEG_MODEL_S3" ]]; then
        info "Auto-detected segmentation model: $SEG_MODEL_S3"
    fi
fi

if [[ -z "$CLASS_MODEL_S3" ]]; then
    CLASS_MODEL_S3=$(aws s3 ls s3://medml-dashboard/model-output/classification/ --region "$REGION" | grep "classification" | sort | tail -1 | awk '{print "s3://medml-dashboard/model-output/classification/" $NF "output/model.tar.gz"}' 2>/dev/null || echo "")
    if [[ -n "$CLASS_MODEL_S3" ]]; then
        info "Auto-detected classification model: $CLASS_MODEL_S3"
    fi
fi

echo ""
echo "============================================================"
echo "  MedML Integration Test Pipeline"
echo "============================================================"
echo "  Region:           $REGION"
echo "  Role:             $ROLE_ARN"
echo "  Segmentation:     $([[ $RUN_SEG == true ]] && echo 'YES' || echo 'SKIP')"
echo "  Classification:   $([[ $RUN_CLASS == true ]] && echo 'YES' || echo 'SKIP')"
echo "  Skip deploy:      $SKIP_DEPLOY"
echo "  Cleanup after:    $CLEANUP"
echo "============================================================"
echo ""

# ── Segmentation Tests ────────────────────────────────────────────────────────
if [[ "$RUN_SEG" == true ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  SEGMENTATION ENDPOINT TEST"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ "$SKIP_DEPLOY" == false ]]; then
        if [[ -z "$SEG_MODEL_S3" ]]; then
            fail "No segmentation model found. Set SEG_MODEL_S3 env var."
        else
            # Repackage model
            info "Repackaging model with inference code..."
            python repackage_model.py --model-s3-uri "$SEG_MODEL_S3" && pass "Model repackaged" || fail "Model repackage failed"

            # Deploy
            info "Deploying segmentation endpoint..."
            python deploy_and_test_segmentation.py \
                --endpoint "$SEG_ENDPOINT" \
                --model-data "$SEG_MODEL_S3" \
                --role-arn "$ROLE_ARN" \
                --test-image "$SEG_TEST_IMAGE" \
                --skip-test && pass "Segmentation endpoint deployed" || fail "Segmentation deployment failed"
        fi
    fi

    # Test inference
    info "Testing segmentation inference..."
    RESULT=$(python deploy_and_test_segmentation.py \
        --endpoint "$SEG_ENDPOINT" \
        --test-image "$SEG_TEST_IMAGE" \
        --skip-deploy 2>&1) || true

    if echo "$RESULT" | grep -q "Segmentation successful"; then
        pass "Segmentation inference returned results"
    elif echo "$RESULT" | grep -q "Result received"; then
        pass "Segmentation async result received"
    else
        fail "Segmentation inference failed or timed out"
        echo "$RESULT" | tail -5
    fi

    # Check mask saved
    MASK_CHECK=$(aws s3 ls s3://medml-dashboard/model-output/segmentation/masks/ --region "$REGION" 2>/dev/null | grep "_segmentation.nii.gz" | tail -1 || echo "")
    if [[ -n "$MASK_CHECK" ]]; then
        pass "Segmentation mask (.nii.gz) saved to S3"
    else
        fail "Segmentation mask not found in S3 output"
    fi
fi

# ── Classification Tests ──────────────────────────────────────────────────────
if [[ "$RUN_CLASS" == true ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  CLASSIFICATION ENDPOINT TEST"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ "$SKIP_DEPLOY" == false ]]; then
        if [[ -z "$CLASS_MODEL_S3" ]]; then
            fail "No classification model found. Set CLASS_MODEL_S3 env var."
        else
            info "Deploying classification endpoint..."
            # Use the test_endpoint.py for classification
            python test_endpoint.py --endpoint "$CLASS_ENDPOINT" 2>&1 | tail -5 || true
        fi
    fi

    # Test real-time inference
    info "Testing classification inference..."
    CLASS_RESULT=$(aws sagemaker-runtime invoke-endpoint \
        --endpoint-name "$CLASS_ENDPOINT" \
        --content-type application/json \
        --body "{\"s3_uri\": \"$CLASS_TEST_IMAGE\"}" \
        --region "$REGION" \
        /tmp/class-integration-output.json 2>&1) || true

    if [[ -f /tmp/class-integration-output.json ]]; then
        CLASS_OUTPUT=$(cat /tmp/class-integration-output.json)
        if echo "$CLASS_OUTPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'predicted_class' in d or 'predictions' in d" 2>/dev/null; then
            pass "Classification inference returned predictions"
            PRED=$(echo "$CLASS_OUTPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('predicted_class', d.get('predictions','?')))" 2>/dev/null)
            info "Prediction: $PRED"
        else
            fail "Classification response missing predictions"
            echo "$CLASS_OUTPUT" | head -3
        fi
        rm -f /tmp/class-integration-output.json
    else
        fail "Classification invoke-endpoint failed"
        echo "$CLASS_RESULT" | tail -3
    fi
fi

# ── Cleanup ───────────────────────────────────────────────────────────────────
if [[ "$CLEANUP" == true && "$SKIP_DEPLOY" == false ]]; then
    echo ""
    info "Cleaning up test endpoints..."
    if [[ "$RUN_SEG" == true ]]; then
        aws sagemaker delete-endpoint --endpoint-name "$SEG_ENDPOINT" --region "$REGION" 2>/dev/null && info "Deleted $SEG_ENDPOINT" || true
        aws sagemaker delete-endpoint-config --endpoint-config-name "$SEG_ENDPOINT-config" --region "$REGION" 2>/dev/null || true
        aws sagemaker delete-model --model-name "$SEG_ENDPOINT-model" --region "$REGION" 2>/dev/null || true
    fi
    if [[ "$RUN_CLASS" == true ]]; then
        aws sagemaker delete-endpoint --endpoint-name "$CLASS_ENDPOINT" --region "$REGION" 2>/dev/null && info "Deleted $CLASS_ENDPOINT" || true
        aws sagemaker delete-endpoint-config --endpoint-config-name "$CLASS_ENDPOINT-config" --region "$REGION" 2>/dev/null || true
        aws sagemaker delete-model --model-name "$CLASS_ENDPOINT-model" --region "$REGION" 2>/dev/null || true
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Test Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
echo "============================================================"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
