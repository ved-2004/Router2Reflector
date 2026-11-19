#!/bin/bash
# Build colab/r2r_colab_bundle.zip for R2R_RAG_Colab.ipynb (run from anywhere).
# Contents: src/, data/raw/, rag_eval/ (code + results), outputs/pipeline_runs/ (direct-pass CSVs for the analysis).
set -e
cd "$(dirname "$0")/../.."          # parent of Router2Reflector/
rm -f Router2Reflector/colab/r2r_colab_bundle.zip
zip -qr Router2Reflector/colab/r2r_colab_bundle.zip \
  Router2Reflector/src Router2Reflector/data/raw Router2Reflector/rag_eval Router2Reflector/outputs/pipeline_runs \
  -x "*/__pycache__/*" "*.pyc" "*/.DS_Store"
ls -lh Router2Reflector/colab/r2r_colab_bundle.zip
