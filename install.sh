#!/bin/sh
# Install banach and banach-engine from the latest GitHub release.
#
#   curl -fsSL https://raw.githubusercontent.com/gautam1858/banach/main/install.sh | sh
#
# Environment: BANACH_VERSION (default: latest), BANACH_INSTALL_DIR (default: ~/.local/bin),
# BANACH_RELEASE_URL (override the download location, for mirrors and tests).
set -eu

repo="gautam1858/banach"
dir="${BANACH_INSTALL_DIR:-$HOME/.local/bin}"
version="${BANACH_VERSION:-latest}"

os=$(uname -s); arch=$(uname -m)
case "$os-$arch" in
  Linux-x86_64)          target=x86_64-unknown-linux-gnu ;;
  Linux-aarch64|Linux-arm64) target=aarch64-unknown-linux-gnu ;;
  Darwin-arm64)          target=aarch64-apple-darwin ;;
  *) echo "banach: no prebuilt binaries for $os $arch; build from source: https://github.com/$repo#build-from-source" >&2; exit 1 ;;
esac

if [ -n "${BANACH_RELEASE_URL:-}" ]; then
  base="$BANACH_RELEASE_URL"
elif [ "$version" = latest ]; then
  base="https://github.com/$repo/releases/latest/download"
else
  base="https://github.com/$repo/releases/download/$version"
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fetch() {
  if command -v curl >/dev/null 2>&1; then curl -fsSL "$1" -o "$2"
  else wget -qO "$2" "$1"; fi
}
echo "downloading banach ($target) ..."
fetch "$base/banach-$target.tar.gz" "$tmp/b.tar.gz"
fetch "$base/banach-$target.tar.gz.sha256" "$tmp/b.sha256"

want=$(cut -d' ' -f1 < "$tmp/b.sha256")
if command -v sha256sum >/dev/null 2>&1; then got=$(sha256sum "$tmp/b.tar.gz" | cut -d' ' -f1)
else got=$(shasum -a 256 "$tmp/b.tar.gz" | cut -d' ' -f1); fi
if [ "$want" != "$got" ]; then
  echo "banach: checksum mismatch (expected $want, got $got)" >&2
  exit 1
fi

tar -xzf "$tmp/b.tar.gz" -C "$tmp"
mkdir -p "$dir"
install -m 755 "$tmp/banach" "$dir/banach"
install -m 755 "$tmp/banach-engine" "$dir/banach-engine"
echo "installed banach and banach-engine to $dir"
case ":$PATH:" in
  *":$dir:"*) ;;
  *) echo "add $dir to your PATH, e.g.: export PATH=\"$dir:\$PATH\"" ;;
esac
echo "try: banach plan hf:unsloth/DeepSeek-V4-Flash-GGUF/UD-Q4_K_XL/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf"
