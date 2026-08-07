#!/usr/bin/env bash
#
# Build ExecuTorch from source and install it where this gem can find it.
#
# Usage:
#   script/build-executorch.sh
#   EXECUTORCH_BACKENDS=xnnpack script/build-executorch.sh
#
# Environment:
#   EXECUTORCH_VERSION   git tag to build           (default: v0.7.0)
#   EXECUTORCH_SRC       where to clone the source  (default: ./tmp/executorch-src/executorch)
#   INSTALL_PREFIX       where to install           (default: ./vendor/executorch)
#   EXECUTORCH_BACKENDS  comma-separated: xnnpack   (default: none)
#   JOBS                 parallel build jobs        (default: detected)
#
# Afterwards:
#   bundle config set --local build.executorch --with-executorch-dir="$PWD/vendor/executorch"
#   bundle exec rake compile

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Single source of truth for the version, shared with CI's cache key.
EXECUTORCH_VERSION="${EXECUTORCH_VERSION:-$(cat "$REPO_ROOT/.executorch-version")}"
EXECUTORCH_SRC="${EXECUTORCH_SRC:-$REPO_ROOT/tmp/executorch-src/executorch}"
INSTALL_PREFIX="${INSTALL_PREFIX:-$REPO_ROOT/vendor/executorch}"
EXECUTORCH_BACKENDS="${EXECUTORCH_BACKENDS:-none}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

BUILD_DIR="$EXECUTORCH_SRC/cmake-out"

log() { printf '\n==> %s\n' "$*"; }

# -----------------------------------------------------------------------------
# Prerequisites
# -----------------------------------------------------------------------------

# ExecuTorch's CMakeLists requires the checkout to be in a directory named
# exactly "executorch" (pytorch/executorch#6475).
if [ "$(basename "$EXECUTORCH_SRC")" != "executorch" ]; then
  echo "EXECUTORCH_SRC must end in a directory named 'executorch', got: $EXECUTORCH_SRC" >&2
  exit 1
fi

cmake_version="$(cmake --version | head -1 | awk '{print $3}')"
required_cmake="3.29"
if [ "$(printf '%s\n%s\n' "$required_cmake" "$cmake_version" | sort -V | head -1)" != "$required_cmake" ]; then
  echo "ExecuTorch needs CMake >= $required_cmake, found $cmake_version." >&2
  echo "Install a newer one, e.g.: pip install 'cmake>=3.29'" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Source
# -----------------------------------------------------------------------------

if [ ! -d "$EXECUTORCH_SRC/.git" ]; then
  log "Cloning ExecuTorch $EXECUTORCH_VERSION"
  mkdir -p "$(dirname "$EXECUTORCH_SRC")"
  git clone --depth 1 --branch "$EXECUTORCH_VERSION" \
    https://github.com/pytorch/executorch.git "$EXECUTORCH_SRC"
fi

log "Fetching submodules (this is the slow part on a cold checkout)"
git -C "$EXECUTORCH_SRC" submodule update --init --recursive --depth 1 --jobs "$JOBS"

# ExecuTorch's CMake shells out to buck2 to generate its source lists, and the
# version has to match the pin in the checkout.
if ! command -v buck2 >/dev/null 2>&1; then
  buck2_pin="$(cat "$EXECUTORCH_SRC/.ci/docker/ci_commit_pins/buck2.txt")"
  log "Installing buck2 $buck2_pin"

  case "$(uname -s)" in
    Darwin) buck2_arch="aarch64-apple-darwin" ;;
    *)      buck2_arch="x86_64-unknown-linux-musl" ;;
  esac

  buck2_dir="$REPO_ROOT/tmp/bin"
  mkdir -p "$buck2_dir"
  curl -sSL -o "$buck2_dir/buck2.zst" \
    "https://github.com/facebook/buck2/releases/download/${buck2_pin}/buck2-${buck2_arch}.zst"

  if command -v zstd >/dev/null 2>&1; then
    zstd -d -f "$buck2_dir/buck2.zst" -o "$buck2_dir/buck2"
  else
    python3 -c "
import sys, zstd
open(sys.argv[2], 'wb').write(zstd.decompress(open(sys.argv[1], 'rb').read()))
" "$buck2_dir/buck2.zst" "$buck2_dir/buck2"
  fi

  chmod +x "$buck2_dir/buck2"
  export PATH="$buck2_dir:$PATH"
fi

# -----------------------------------------------------------------------------
# Configure and build
# -----------------------------------------------------------------------------

cmake_args=(
  -B "$BUILD_DIR"
  -S "$EXECUTORCH_SRC"
  "-DBUCK2=$(command -v buck2)"
  "-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON
  -DEXECUTORCH_BUILD_PTHREADPOOL=ON
  -DEXECUTORCH_BUILD_CPUINFO=ON
)

if [[ ",$EXECUTORCH_BACKENDS," == *",xnnpack,"* ]]; then
  # Worth ~658x on resnet18 -- see bench/FINDINGS.md. Costs a much longer build,
  # and models must also be exported through XnnpackPartitioner to benefit.
  log "Including the XNNPACK delegate"
  cmake_args+=(-DEXECUTORCH_BUILD_XNNPACK=ON)
fi

command -v ninja >/dev/null 2>&1 && cmake_args+=(-G Ninja)

log "Configuring"
cmake "${cmake_args[@]}"

log "Building with $JOBS jobs"
cmake --build "$BUILD_DIR" -j "$JOBS"

log "Installing to $INSTALL_PREFIX"
cmake --install "$BUILD_DIR"

# -----------------------------------------------------------------------------
# Headers
# -----------------------------------------------------------------------------
#
# `cmake --install` only installs a subset of ExecuTorch's headers -- notably
# not extension/module/module.h or anything under runtime/executor, both of
# which this gem includes directly. Without these, `rake compile` fails with
# "module.h header not found" even though the install looks complete.
#
# Copy the rest out of the source tree, preserving the layout the headers
# #include each other by.

log "Copying headers the install step leaves behind"
header_root="$INSTALL_PREFIX/include/executorch"
copied=0
while IFS= read -r header; do
  dest="$header_root/$header"
  mkdir -p "$(dirname "$dest")"
  cp "$EXECUTORCH_SRC/$header" "$dest"
  copied=$((copied + 1))
done < <(cd "$EXECUTORCH_SRC" && find runtime kernels extension \
           -name '*.h' -not -path '*/test/*' -not -path '*/testing_util/*')

log "Copied $copied headers"

if [ ! -f "$header_root/extension/module/module.h" ]; then
  echo "Expected $header_root/extension/module/module.h to exist; build looks incomplete." >&2
  exit 1
fi

cat <<EOF

ExecuTorch $EXECUTORCH_VERSION installed to $INSTALL_PREFIX

Next:
  bundle config set --local build.executorch --with-executorch-dir="$INSTALL_PREFIX"
  bundle exec rake compile
  bundle exec rake test
EOF
