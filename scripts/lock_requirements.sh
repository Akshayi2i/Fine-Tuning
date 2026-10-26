#!/usr/bin/env bash
# Regenerate the lock file for each pod environment, and the Azure service's image.
#
#   bash scripts/lock_requirements.sh          (needs uv: pip install uv)
#
# pyproject.toml declares ranges; the requirements-*.txt files choose groups per
# pod; the requirements-*.lock files pin every package, transitive ones included,
# to the versions resolved for the pod: Linux x86-64, Python 3.11. setup_pod.sh
# installs each role's requirements file constrained by its lock, so a pod set up
# next month gets the same versions as one set up today.
#
# Re-run after changing a dependency in pyproject.toml, and commit the locks.
# tests/test_dependencies.py fails when a lock no longer satisfies pyproject.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null 2>&1 || { echo "lock_requirements: needs uv (pip install uv)" >&2; exit 1; }

for role in train ocr serve quantize; do
  # Serving runs exactly the versions calibration was fitted with: its lock is
  # resolved inside the training lock (pillow resizes the pages the model sees).
  constraint=()
  [ "$role" = serve ] && constraint=(-c requirements-train.lock)
  uv pip compile "requirements-$role.txt" "${constraint[@]}" \
    --python-platform x86_64-manylinux_2_28 --python-version 3.11 \
    --no-emit-package insurance-extraction-finetuning \
    --no-header --quiet \
    -o "requirements-$role.lock"
  echo "requirements-$role.lock: $(grep -c '==' "requirements-$role.lock") packages"
done

# The Azure service's image (azure_service/Dockerfile): no torch, so not inside the
# training lock as a whole - but its pillow and PyMuPDF are the training lock's,
# because the worker renders the pages the model reads (one renderer, one set of pixels).
# A relative name: the lock's "via" notes name the constraint file, and must name no machine's path.
render_pins=".render-pins.txt"
trap 'rm -f "$render_pins"' EXIT
grep -iE '^(pillow|pymupdf)==' requirements-train.lock > "$render_pins"
uv pip compile requirements-service.txt -c "$render_pins"   --python-platform x86_64-manylinux_2_28 --python-version 3.11   --no-emit-package insurance-extraction-finetuning   --no-header --quiet   -o requirements-service.lock
echo "requirements-service.lock: $(grep -c '==' requirements-service.lock) packages"
