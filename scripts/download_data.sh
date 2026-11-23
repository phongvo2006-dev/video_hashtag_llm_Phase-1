#!/usr/bin/env bash
# Downloads the MSR-VTT annotations (captions + categories, ~4 MB zip, no videos needed).
# Text processing uses TensorFlow, so no extra language resources are required.
# Source: the CLIP4Clip release mirror of the official MSR-VTT 2017 annotation file.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
if [ ! -f data/raw/msrvtt_data/MSRVTT_data.json ]; then
  curl -L -o data/raw/msrvtt_data.zip \
    https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
  (cd data/raw && unzip -o -q msrvtt_data.zip && rm msrvtt_data.zip)
fi
echo "MSR-VTT annotations ready in data/raw/msrvtt_data/"
