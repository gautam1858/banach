#!/usr/bin/env node
// Launcher: finds the prebuilt binaries in the platform package npm installed for this
// machine (banach-<platform>-<arch>) and runs them.
'use strict';
const { spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const PLATFORMS = {
  'linux-x64': 'banach-linux-x64',
  'linux-arm64': 'banach-linux-arm64',
  'darwin-arm64': 'banach-darwin-arm64',
};
const key = `${process.platform}-${process.arch}`;
const pkg = PLATFORMS[key];
if (!pkg) {
  console.error(`banach: no prebuilt binaries for ${key}. Build from source: https://github.com/gautam1858/banach#build-from-source`);
  process.exit(1);
}
let dir;
try {
  dir = path.dirname(require.resolve(`${pkg}/package.json`));
} catch (e) {
  console.error(`banach: the ${pkg} package is not installed. Reinstall without --omit=optional / --no-optional.`);
  process.exit(1);
}
const exe = path.join(dir, 'bin', 'banach');
const engine = path.join(dir, 'bin', 'banach-engine');
const env = { ...process.env };
if (!env.BANACH_ENGINE && fs.existsSync(engine)) env.BANACH_ENGINE = engine;
const r = spawnSync(exe, process.argv.slice(2), { stdio: 'inherit', env });
if (r.error) {
  console.error(`banach: ${r.error.message}`);
  process.exit(1);
}
process.exit(r.status === null ? 1 : r.status);
