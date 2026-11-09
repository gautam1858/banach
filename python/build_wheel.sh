#!/usr/bin/env bash
# Build a platform wheel around prebuilt binaries.
# Usage: build_wheel.sh BANACH_BIN ENGINE_BIN PLATFORM_TAG OUT_DIR
#   PLATFORM_TAG e.g. manylinux_2_28_x86_64, manylinux_2_28_aarch64, macosx_14_0_arm64
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
banach=$1; engine=$2; tag=$3; out=$4
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp -r "$here/." "$work/"
# PyPI renders the README away from the repository: relative images and links become absolute
python3 "$here/../scripts/pypi_readme.py" "$here/../README.md" > "$work/README.md"
cp "$here/../LICENSE" "$here/../THIRD_PARTY_NOTICES.md" "$work/"
mkdir -p "$work/src/banach/bin"
install -m 755 "$banach" "$work/src/banach/bin/banach"
install -m 755 "$engine" "$work/src/banach/bin/banach-engine"
python3 -m pip wheel --no-deps --no-build-isolation -w "$work/dist" "$work" >/dev/null
whl=$(ls "$work"/dist/banach-*.whl)
python3 -m wheel tags --remove --python-tag py3 --abi-tag none --platform-tag "$tag" "$whl" >/dev/null
mkdir -p "$out"
cp "$work"/dist/banach-*-py3-none-"$tag".whl "$out/"
ls "$out"
