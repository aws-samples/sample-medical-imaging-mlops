#!/bin/bash
# Sync this project to a remote host over SSH via rsync.
#
# Excludes virtualenvs, caches, git internals, and secrets (.env.local, *.pem).
#
# Usage:
#   ./scripts/sync-ec2.sh                 # sync to the default host
#   ./scripts/sync-ec2.sh --dry-run       # preview what would transfer
#   ./scripts/sync-ec2.sh --delete        # also remove remote files deleted locally
#   ./scripts/sync-ec2.sh --dry-run --delete
#
# Override target via environment variables:
#   REMOTE_HOST=ubuntu@host.example.com \
#   SSH_KEY=~/Documents/AWS/vikash-mac.pem \
#   REMOTE_DIR='~/medical-image-training-dashboard/' \
#   ./scripts/sync-ec2.sh

set -euo pipefail

# ── Config (override via env) ────────────────────────────────────────────────
REMOTE_HOST="${REMOTE_HOST:-ubuntu@ec2-3-141-147-16.us-east-2.compute.amazonaws.com}"
SSH_KEY="${SSH_KEY:-$HOME/Documents/AWS/vikash-mac.pem}"
REMOTE_DIR="${REMOTE_DIR:-~/medical-image-training-dashboard/}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # project root
LOCAL_DIR="$SCRIPT_DIR/"

# ── Flags ────────────────────────────────────────────────────────────────────
RSYNC_OPTS=()

for arg in "$@"; do
    case "$arg" in
        --dry-run) RSYNC_OPTS+=(-n) ;;
        --delete)  RSYNC_OPTS+=(--delete) ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ── Preflight ────────────────────────────────────────────────────────────────
if [[ ! -f "$SSH_KEY" ]]; then
    echo "❌ SSH key not found: $SSH_KEY"
    echo "   Set SSH_KEY=/path/to/key.pem or place the key at the default path."
    exit 1
fi

# SSH refuses keys that are group/world-readable.
KEY_PERMS="$(stat -f '%Lp' "$SSH_KEY" 2>/dev/null || stat -c '%a' "$SSH_KEY" 2>/dev/null || echo '')"
if [[ -n "$KEY_PERMS" && "$KEY_PERMS" != "400" && "$KEY_PERMS" != "600" ]]; then
    echo "⚠️  SSH key perms are $KEY_PERMS; tightening to 600."
    chmod 600 "$SSH_KEY"
fi

SSH_OPTS="ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"

echo "Syncing:  $LOCAL_DIR"
echo "      ->  $REMOTE_HOST:$REMOTE_DIR"
[[ " ${RSYNC_OPTS[*]-} " == *" -n "* ]] && echo "Mode:     DRY RUN (no changes)"
[[ " ${RSYNC_OPTS[*]-} " == *" --delete "* ]] && echo "Delete:   ON (remote files removed if deleted locally)"

# shellcheck disable=SC2086
rsync -az --stats ${RSYNC_OPTS[@]+"${RSYNC_OPTS[@]}"} \
    -e "$SSH_OPTS" \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    --exclude='.git/' \
    --exclude='.DS_Store' \
    --exclude='.env.local' \
    --exclude='*.pem' \
    --exclude='logs/*.out' \
    "$LOCAL_DIR" "$REMOTE_HOST:$REMOTE_DIR"

echo "✅ Done."
