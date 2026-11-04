# Releasing

A release is one tag. `.github/workflows/release.yml` builds Linux x86_64, Linux arm64 and
macOS arm64 binaries, then publishes to GitHub Releases, PyPI, npm, crates.io and GHCR.

## One-time setup

1. **PyPI** (no secret): on pypi.org, Account settings, Publishing, add a pending trusted
   publisher: project `banach`, owner `gautam1858`, repository `banach`, workflow
   `release.yml`, environment `pypi`. In the GitHub repository create an environment named
   `pypi` (Settings, Environments).
2. **npm**: create an automation token on npmjs.com and add it as the repository secret
   `NPM_TOKEN` (Settings, Secrets and variables, Actions). The packages published are
   `banach`, `banach-linux-x64`, `banach-linux-arm64` and `banach-darwin-arm64`.
3. **crates.io**: create an API token on crates.io and add it as the repository secret
   `CARGO_REGISTRY_TOKEN`.
4. **GHCR**: nothing to do; the workflow uses the built-in `GITHUB_TOKEN`. After the first
   push, set the package `banach` to public under your GitHub profile's Packages.

Registries whose secret is missing are skipped, so a release can go out to fewer places.

## Each release

1. Bump the version in `Cargo.toml`, `python/pyproject.toml`, `python/src/banach/__init__.py`,
   `npm/banach/package.json` (version and the three `optionalDependencies`).
2. Commit, then tag and push:

   ```bash
   git tag v0.1.0 && git push origin v0.1.0
   ```

3. Check the run under Actions. The release page gets the tarballs and checksums that
   `install.sh` downloads.

## Inline video on the README

GitHub plays a video inline only when it is uploaded through the web editor. To add the
film as a player: edit README.md on github.com, drag `brand/video/banach-launch-16x9.mp4`
into the editor where the player should go, and commit. GitHub replaces it with a
`https://github.com/user-attachments/assets/...` link that renders as a player. The GIF at
the top of the README needs nothing: it plays as soon as the page opens.
