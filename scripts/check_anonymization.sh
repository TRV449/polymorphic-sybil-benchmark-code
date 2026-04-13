#!/usr/bin/env bash
# =============================================================================
# check_anonymization.sh — Double-blind submission violation scanner
#
# Scans all tracked code/docs for identifying information:
#   - Author names, emails, student IDs
#   - Institution names
#   - Git config leaks
#   - HF/GitHub URLs with real usernames
#
# Usage:
#   bash scripts/check_anonymization.sh
# =============================================================================
set -euo pipefail

ROOT="${WORKSPACE_ROOT:-/mnt/data/2020112002}"
VIOLATIONS=0

red()   { echo -e "\033[31m$*\033[0m"; }
green() { echo -e "\033[32m$*\033[0m"; }
yellow(){ echo -e "\033[33m$*\033[0m"; }

check_section() {
    local title=$1
    local count=$2
    if [ "$count" -gt 0 ]; then
        red "  ✗ $title: $count match(es) found"
        VIOLATIONS=$((VIOLATIONS + count))
    else
        green "  ✓ $title: clean"
    fi
}

echo "================================================================="
echo "  Anonymization Check — NeurIPS 2026 E&D Submission"
echo "  Root: $ROOT"
echo "================================================================="

# Common grep options: tracked files only, exclude noise
GREP_OPTS="-rn --include=*.py --include=*.md --include=*.sh --include=*.json --include=*.yaml --include=*.yml --include=*.txt --include=*.tex --exclude=check_anonymization.sh --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=.venv-awq --exclude-dir=llm_cache --exclude-dir=_not_used --exclude-dir=_indexes --exclude-dir=_pyserini_input --exclude-dir=grid_search_runs --exclude-dir=paper_results --exclude-dir=fair_v1_logs_20260309_1225 --exclude-dir=.Trash-0 --exclude-dir=_non_benchmark --exclude-dir=siyeoncode --exclude-dir=models --exclude-dir=GMTP --exclude-dir=Mask_DOC --exclude-dir=llama.cpp --exclude-dir=llama-3.1 --exclude-dir=.cache --exclude-dir=member_runtime --exclude-dir=gpt-oss-120b-GGUF --exclude-dir=wiki_indexes --exclude-dir=faiss_index_wikipedia_dpr_100w --exclude-dir=scripts"

# ── 1. Author real names ────────────────────────────────────────────────────
echo ""
echo "[1] Author Real Names"
MATCHES=$(grep $GREP_OPTS -i -c \
    -e "sujeong" -e "siyeon" -e "donghyun" -e "dongyun" \
    -e "이수정" -e "이시연" -e "이동현" -e "이동윤" \
    -e "choi.su" -e "lee.dong" \
    "$ROOT" 2>/dev/null | awk -F: '{s+=$2}END{print s+0}')
check_section "Author names" "$MATCHES"
if [ "$MATCHES" -gt 0 ]; then
    grep $GREP_OPTS -i \
        -e "sujeong" -e "siyeon" -e "donghyun" -e "dongyun" \
        -e "이수정" -e "이시연" -e "이동현" -e "이동윤" \
        -e "choi.su" -e "lee.dong" \
        "$ROOT" 2>/dev/null | grep -v "LICENSE" | head -10
fi

# ── 2. Email addresses ──────────────────────────────────────────────────────
echo ""
echo "[2] Email Addresses"
MATCHES=$(grep $GREP_OPTS -E -c \
    "[a-zA-Z0-9._+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" \
    "$ROOT" 2>/dev/null | awk -F: '{s+=$2}END{print s+0}')
# Filter out safe emails
REAL_MATCHES=$(grep $GREP_OPTS -E \
    "[a-zA-Z0-9._+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" \
    "$ROOT" 2>/dev/null | \
    grep -v "anonymous@neurips2026" | \
    grep -v "noreply@" | \
    grep -v "example\.com" | \
    grep -v "LICENSE" | \
    grep -v "NOTICE" | wc -l)
check_section "Email addresses (non-anonymous)" "$REAL_MATCHES"
if [ "$REAL_MATCHES" -gt 0 ]; then
    grep $GREP_OPTS -E \
        "[a-zA-Z0-9._+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" \
        "$ROOT" 2>/dev/null | \
        grep -v "anonymous@neurips2026" | \
        grep -v "noreply@" | \
        grep -v "example\.com" | \
        grep -v "LICENSE" | \
        grep -v "NOTICE" | head -10
fi

# ── 3. Student ID / workspace path ──────────────────────────────────────────
echo ""
echo "[3] Student ID / Identifying Path"
MATCHES=$(grep $GREP_OPTS -c \
    -e "2020112002" -e "shalalastal" -e "dongyun0215" \
    "$ROOT" 2>/dev/null | awk -F: '{s+=$2}END{print s+0}')
check_section "Student ID / personal handles" "$MATCHES"
if [ "$MATCHES" -gt 0 ]; then
    grep $GREP_OPTS \
        -e "2020112002" -e "shalalastal" -e "dongyun0215" \
        "$ROOT" 2>/dev/null | head -10
fi

# ── 4. Institution names ────────────────────────────────────────────────────
echo ""
echo "[4] Institution Names"
MATCHES=$(grep $GREP_OPTS -i -c \
    -e "dong-a" -e "kookmin" -e "sogang" -e "yonsei" \
    -e "kaist" -e "postech" -e "snu" -e "korea univ" \
    -e "동아대" -e "국민대" -e "서강대" -e "연세대" \
    "$ROOT" 2>/dev/null | awk -F: '{s+=$2}END{print s+0}')
check_section "Institution names" "$MATCHES"
if [ "$MATCHES" -gt 0 ]; then
    grep $GREP_OPTS -i \
        -e "dong-a" -e "kookmin" -e "sogang" -e "yonsei" \
        -e "kaist" -e "postech" -e "snu" -e "korea univ" \
        -e "동아대" -e "국민대" -e "서강대" -e "연세대" \
        "$ROOT" 2>/dev/null | head -10
fi

# ── 5. Copyright with real names ────────────────────────────────────────────
echo ""
echo "[5] Copyright Notices"
MATCHES=$(grep $GREP_OPTS -i -c \
    -e "copyright" -e "©" \
    "$ROOT" 2>/dev/null | awk -F: '{s+=$2}END{print s+0}')
REAL_MATCHES=$(grep $GREP_OPTS -i \
    -e "copyright" -e "©" \
    "$ROOT" 2>/dev/null | \
    grep -v "Anonymous" | \
    grep -v "LICENSE" | \
    grep -v "NOTICE" | wc -l)
check_section "Copyright (non-anonymous)" "$REAL_MATCHES"
if [ "$REAL_MATCHES" -gt 0 ]; then
    grep $GREP_OPTS -i \
        -e "copyright" -e "©" \
        "$ROOT" 2>/dev/null | \
        grep -v "Anonymous" | \
        grep -v "LICENSE" | \
        grep -v "NOTICE" | head -10
fi

# ── 6. Git config ───────────────────────────────────────────────────────────
echo ""
echo "[6] Git Config"
cd "$ROOT"
GIT_NAME=$(git config user.name 2>/dev/null || echo "(not set)")
GIT_EMAIL=$(git config user.email 2>/dev/null || echo "(not set)")

if echo "$GIT_NAME" | grep -qi "anonymous\|anon"; then
    green "  ✓ user.name: $GIT_NAME"
else
    red "  ✗ user.name: $GIT_NAME (NOT anonymous)"
    VIOLATIONS=$((VIOLATIONS + 1))
fi

if echo "$GIT_EMAIL" | grep -qi "anonymous\|neurips\|invalid"; then
    green "  ✓ user.email: $GIT_EMAIL"
else
    red "  ✗ user.email: $GIT_EMAIL (NOT anonymous)"
    VIOLATIONS=$((VIOLATIONS + 1))
fi

# Check commit history
COMMIT_COUNT=$(git log --all --oneline 2>/dev/null | wc -l)
if [ "$COMMIT_COUNT" -gt 0 ]; then
    echo "  Commit authors:"
    git log --all --format="    %an <%ae>" 2>/dev/null | sort -u
    NON_ANON=$(git log --all --format="%an %ae" 2>/dev/null | grep -iv "anonymous" | wc -l)
    if [ "$NON_ANON" -gt 0 ]; then
        red "  ✗ $NON_ANON non-anonymous commit(s) in history"
        VIOLATIONS=$((VIOLATIONS + NON_ANON))
    fi
else
    green "  ✓ No commits yet (clean history)"
fi

# ── 7. File paths with identifying info ──────────────────────────────────────
echo ""
echo "[7] File Paths"
PATH_MATCHES=$(find "$ROOT" -maxdepth 4 -type f \
    \( -name "*.py" -o -name "*.md" -o -name "*.sh" -o -name "*.json" \) \
    2>/dev/null | grep -i -c \
    -e "sujeong" -e "siyeon" -e "donghyun" -e "choi" || echo 0)
check_section "File paths with names" "$PATH_MATCHES"

# ── 8. HF / GitHub URLs ─────────────────────────────────────────────────────
echo ""
echo "[8] HF / GitHub URLs"
URL_MATCHES=$(grep $GREP_OPTS \
    -e "huggingface.co/" -e "github.com/" \
    "$ROOT" 2>/dev/null | \
    grep -v "colbert-ir\|intfloat\|cross-encoder\|facebook\|google\|openai" | \
    grep -v "LICENSE\|NOTICE" | \
    grep -v "anonymous\|anon" | wc -l)
check_section "Non-model HF/GitHub URLs" "$URL_MATCHES"
if [ "$URL_MATCHES" -gt 0 ]; then
    grep $GREP_OPTS \
        -e "huggingface.co/" -e "github.com/" \
        "$ROOT" 2>/dev/null | \
        grep -v "colbert-ir\|intfloat\|cross-encoder\|facebook\|google\|openai" | \
        grep -v "LICENSE\|NOTICE" | \
        grep -v "anonymous\|anon" | head -10
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================================="
if [ "$VIOLATIONS" -eq 0 ]; then
    green "  PASS: No anonymization violations detected."
else
    red "  FAIL: $VIOLATIONS violation(s) found. Fix before submission."
fi
echo "================================================================="
exit $VIOLATIONS
