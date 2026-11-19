#!/bin/bash
# Build colab/R2R_Colab_Kit.zip: the one file to share with teammates.
#   R2R_Colab_Kit/README.txt, R2R_Colab_Kit/r2r_colab_bundle.zip, R2R_Colab_Kit/notebooks/R2R_RAG_Colab_{3B,7B,14B,32B}.ipynb
# If colab/r2r_med_contexts.zip exists (shared med retrieval from an earlier Colab run), it is folded into the bundle
# so every size uses identical passages; otherwise each notebook rebuilds the med retrieval itself.
set -e
cd "$(dirname "$0")"
if [ -f r2r_med_contexts.zip ]; then
  unzip -qo r2r_med_contexts.zip -d ../rag_eval/results && echo "folded shared med contexts into the bundle"
fi
python3 build_notebook.py >/dev/null
./make_bundle.sh >/dev/null
B=$(mktemp -d); mkdir -p "$B/R2R_Colab_Kit/notebooks"
cp README_TEAM.txt "$B"/R2R_Colab_Kit/README.txt
cp r2r_colab_bundle.zip "$B"/R2R_Colab_Kit/
cp R2R_RAG_Colab_*.ipynb "$B"/R2R_Colab_Kit/notebooks/
rm -f R2R_Colab_Kit.zip
(cd "$B" && zip -qr "$OLDPWD/R2R_Colab_Kit.zip" R2R_Colab_Kit)
rm -rf "$B"
unzip -l R2R_Colab_Kit.zip
