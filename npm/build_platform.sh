#!/usr/bin/env bash
# Assemble the npm platform package banach-<platform> around prebuilt binaries.
# Usage: build_platform.sh PLATFORM BANACH_BIN ENGINE_BIN OUT_DIR
#   PLATFORM one of linux-x64, linux-arm64, darwin-arm64
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
plat=$1; banach=$2; engine=$3; out=$4
os=${plat%-*}; cpu=${plat#*-}
version=$(sed -n 's/^  "version": "\(.*\)",$/\1/p' "$here/banach/package.json")
[ -n "$version" ] || { echo "cannot read the version from banach/package.json" >&2; exit 1; }
dir="$out/banach-$plat"
mkdir -p "$dir/bin"
install -m 755 "$banach" "$dir/bin/banach"
install -m 755 "$engine" "$dir/bin/banach-engine"
cat > "$dir/package.json" <<JSON
{
  "name": "banach-$plat",
  "version": "$version",
  "description": "Prebuilt banach binaries for $plat (installed automatically by the banach package)",
  "license": "Apache-2.0",
  "repository": { "type": "git", "url": "git+https://github.com/gautam1858/banach.git" },
  "os": ["$os"],
  "cpu": ["$cpu"],
  "files": ["bin/"]
}
JSON
echo "$dir"
