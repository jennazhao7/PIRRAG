#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${SCRIPT_DIR}/dist"
STAGE_DIR="${DIST_DIR}/openfhe_single_query_bundle"

echo "[1/4] Preparing staging directory..."
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

echo "[2/4] Copying required files..."
copy_file() {
  local rel="$1"
  mkdir -p "${STAGE_DIR}/$(dirname "${rel}")"
  cp "${REPO_ROOT}/${rel}" "${STAGE_DIR}/${rel}"
}

copy_file "openfhe_core/CMakeLists.txt"
copy_file "openfhe_core/README.md"
copy_file "openfhe_core/include/io_utils.h"
copy_file "openfhe_core/src/io_utils.cpp"
copy_file "openfhe_core/src/openfhe_keygen.cpp"
copy_file "openfhe_core/src/openfhe_encrypt_query.cpp"
copy_file "openfhe_core/src/openfhe_compute_distances.cpp"
copy_file "openfhe_core/src/openfhe_decrypt_topk.cpp"
copy_file "prototype/fhe_backend.py"
copy_file "prototype/fhe_query_client.py"
copy_file "prototype/fhe_query_server.py"

echo "[3/4] Writing bundle metadata..."
cat > "${STAGE_DIR}/BUNDLE_INFO.txt" <<'EOF'
OpenFHE single-query FHE bundle.

Build:
  cmake -S openfhe_core -B openfhe_core/build -DOpenFHE_DIR=/usr/local/lib/OpenFHE
  cmake --build openfhe_core/build -j

Run instructions are documented in openfhe_core/README.md.
EOF

echo "[4/4] Creating tarball..."
mkdir -p "${DIST_DIR}"
tar -czf "${DIST_DIR}/openfhe_single_query_bundle.tar.gz" -C "${DIST_DIR}" "openfhe_single_query_bundle"

echo "Bundle created:"
echo "  ${DIST_DIR}/openfhe_single_query_bundle.tar.gz"
