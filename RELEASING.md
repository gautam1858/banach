# Releasing

Banach is published to PyPI only (`pip install banach`). The GitHub release also carries the
Linux and macOS binary archives with checksums, for people who want the binaries without pip.

## One-time setup

1. **PyPI trusted publisher** (no secret): on pypi.org, Account settings, Publishing, add a
   publisher for project `banach`, owner `gautam1858`, repository `banach`, workflow
   `release.yaml`, environment `pypi`.
2. **Protect the `pypi` environment**: in the GitHub repository, Settings, Environments, `pypi`,
   tick "Required reviewers" and add yourself. A release then waits for your approval before
   anything is uploaded, so a stray workflow run cannot publish.

## Each release

1. Bump the version in `Cargo.toml`, `python/pyproject.toml` and
   `python/src/banach/__init__.py`, run `cargo build --release` so `Cargo.lock` follows, and
   commit and push.
2. On GitHub open Actions, pick "release", press "Run workflow" and type the version (for
   example `0.1.4`). The workflow creates the tag and the release itself. (Pushing a `v*` tag
   works too.)
3. The workflow runs in this order, and stops at the first failure:
   - **check**: the version strings in the three files above match.
   - **build**: Linux x86_64, Linux arm64 and macOS arm64 binaries and wheels. The macOS leg is
     allowed to fail (its runners are sometimes unavailable); the Linux wheels still ship.
   - **smoke**: each Linux wheel is installed in a clean virtual environment, then the
     byte-exact oracle (`engine/exact_check.sh`) and `scripts/smoke_test.sh` run against the
     binaries inside it: `banach compare`, `banach pick`, the server's API key, Host check, CORS
     and content-type rules, and a real completion.
   - **github-release** and **pypi**: only if smoke passed. PyPI waits for the reviewer approval
     from the setup above.
4. Check the PyPI page. It can take a few minutes before `pip install banach==VERSION` finds the
   new version; `pip install --no-cache-dir` helps.

## Notices

`THIRD_PARTY_NOTICES.md` ships in every wheel. Regenerate it when dependencies change, with the
llama.cpp checkout the engine is built against:

```bash
cargo fetch --locked
python3 scripts/gen_third_party_notices.py ../llama.cpp > THIRD_PARTY_NOTICES.md
```

## Inline video on the README

GitHub plays a video inline only when it is uploaded through the web editor. To add the
film as a player: edit README.md on github.com, drag `brand/video/banach-launch-16x9.mp4`
into the editor where the player should go, and commit. GitHub replaces it with a
`https://github.com/user-attachments/assets/...` link that renders as a player. The GIF at
the top of the README needs nothing: it plays as soon as the page opens.
