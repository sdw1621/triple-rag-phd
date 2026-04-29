#!/usr/bin/env bash
# Re-evaluate all policies (per-query dump enabled) and run paired bootstrap.
#
# Run inside the project Docker container, e.g.
#     docker-compose exec triple_rag bash scripts/run_paired_bootstrap.sh
#
# Cost: $0 (cache lookup only, no LLM calls). Wall-clock: ~2 min total.

set -euo pipefail

CACHE=cache/university.sqlite
QA=data/university/gold_qa_5000.json
RESULTS=results

run_eval () {
  local policy="$1"
  local out="$2"
  echo "=== eval $policy → $out ==="
  python scripts/evaluate_on_cache.py \
    --cache "$CACHE" --qa "$QA" \
    --policy "$policy" --output "$out"
}

# 1) Re-evaluate every policy with per-query dump enabled.
run_eval rdwa                                            "$RESULTS/eval_rdwa.json"
run_eval uniform                                         "$RESULTS/eval_uniform.json"
run_eval oracle                                          "$RESULTS/eval_oracle.json"
run_eval vector-only                                     "$RESULTS/eval_vector-only.json"
run_eval graph-only                                      "$RESULTS/eval_graph-only.json"
run_eval ontology-only                                   "$RESULTS/eval_ontology-only.json"
run_eval ldwa:cache/ppo_checkpoints/seed_42/final.pt     "$RESULTS/eval_ldwa_seed42_cache.json"
run_eval ldwa:cache/ppo_checkpoints/seed_123/final.pt    "$RESULTS/eval_ldwa_seed123_cache.json"
run_eval ldwa:cache/ppo_checkpoints/seed_999/final.pt    "$RESULTS/eval_ldwa_seed999_cache.json"

POLICIES_BASE=(
  "rdwa=$RESULTS/eval_rdwa.json"
  "uniform=$RESULTS/eval_uniform.json"
  "vector_only=$RESULTS/eval_vector-only.json"
  "graph_only=$RESULTS/eval_graph-only.json"
  "ontology_only=$RESULTS/eval_ontology-only.json"
  "ldwa_seed42=$RESULTS/eval_ldwa_seed42_cache.json"
  "ldwa_seed123=$RESULTS/eval_ldwa_seed123_cache.json"
  "ldwa_seed999=$RESULTS/eval_ldwa_seed999_cache.json"
  "oracle=$RESULTS/eval_oracle.json"
)

# 2) Paired bootstrap, two viewpoints (Q1: vs R-DWA, Q2: vs Oracle).
for METRIC in f1 faith r; do
  python scripts/paired_bootstrap.py \
    --policies "${POLICIES_BASE[@]}" \
    --baseline rdwa \
    --metric "$METRIC" \
    --output "$RESULTS/paired_bootstrap_${METRIC}_vs_rdwa.md"

  python scripts/paired_bootstrap.py \
    --policies "${POLICIES_BASE[@]}" \
    --baseline oracle \
    --metric "$METRIC" \
    --output "$RESULTS/paired_bootstrap_${METRIC}_vs_oracle.md"
done

echo
echo "=== DONE ==="
echo "Generated:"
ls -1 "$RESULTS"/paired_bootstrap_*.md
