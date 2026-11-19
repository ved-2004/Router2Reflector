#!/bin/bash
# Pull evaluation artifacts from the r2r-rag-eval volume into rag_eval/results/ (run from Router2Reflector/).
set -e
cd "$(dirname "$0")"
for p in retrieval runs gen; do modal volume get --force r2r-rag-eval /$p results/ ; done
for f in nli.json contam.json; do modal volume get --force r2r-rag-eval /$f results/$f || true; done
