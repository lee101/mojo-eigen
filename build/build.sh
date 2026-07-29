#!/usr/bin/env bash
set -euo pipefail

mkdir -p dist
mojo build --emit shared-lib src/eigen.mojo -o dist/libmojo-eigen.so -Xlinker -lm
