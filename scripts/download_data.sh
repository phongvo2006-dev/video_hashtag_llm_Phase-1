#!/usr/bin/env bash
# Downloads the MSR-VTT annotations (captions + categories, ~4 MB zip, no videos needed).
# Source: the CLIP4Clip release mirror of the official MSR-VTT 2017 annotation file.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dataset
if [ ! -f dataset/MSRVTT_data.json ]; then
  curl -L -o /tmp/msrvtt_data.zip \
    https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
  tmp_extract="$(mktemp -d)"
  unzip -o -q /tmp/msrvtt_data.zip -d "$tmp_extract"
  mv "$tmp_extract"/msrvtt_data/* dataset/
  rm -rf /tmp/msrvtt_data.zip "$tmp_extract"
fi
python -c "import nltk; [nltk.download(p, quiet=True) for p in ['wordnet','omw-1.4','averaged_perceptron_tagger_eng']]"
echo "MSR-VTT annotations ready in dataset/"
