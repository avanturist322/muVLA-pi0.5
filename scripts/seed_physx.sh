#!/usr/bin/env bash
# Put SAPIEN's GPU PhysX library on NFS, once, from a single process.
#
#     bash scripts/seed_physx.sh
#
# Why this exists. sapien.physx.enable_gpu() lazily downloads libPhysXGpu_64.so (236 MB)
# into `Path.home()/.sapien/physx/<version>`. That path is hardcoded to Path.home() - the
# SAPIEN_PHYSX_CACHE_ROOT variable does not redirect it, it controls the PhysX *kernel*
# cache, which is a different thing. On the cluster this was built for $HOME is ephemeral,
# so the library is missing on every new job, and the eval suite starts one process per
# GPU at the same moment. All of them find it missing, all of them download it into the
# same path, and the ones that lose the race dlopen a partially written file:
#
#     OSError: .../libPhysXGpu_64.so: file too short     (exit 1)
#     Bus error                                          (exit -7, SIGBUS on the mmap)
#
# On 2026-07-26 that destroyed 12 of 23 envs in one suite. Downloading once, here, into
# NFS - and symlinking $HOME at it in scripts/eval_env.sh - removes the download from the
# hot path, and with it the race.
#
# Idempotent: it is a no-op when the cached file is already complete.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
CACHE="$ROOT/.cache/sapien/physx-lib"

# Ask the installed sapien which version it wants rather than hardcoding one: the URL and
# the directory name are both derived from it, and a mismatch would silently re-download.
VERSION="$("$PY" -c 'import sapien.physx as p; print(p.version())' 2>/dev/null | tail -n 1)"
if [ -z "$VERSION" ]; then
    echo "seed_physx: could not import sapien.physx with $PY" >&2
    exit 1
fi
DEST="$CACHE/$VERSION"
SO="$DEST/libPhysXGpu_64.so"

# 236 MB is the size of the known-good file; anything much smaller is a truncated
# download, which is exactly the failure mode this script exists to prevent - so treat it
# as absent rather than trusting it.
MIN_BYTES=200000000
if [ -f "$SO" ] && [ "$(stat -c %s "$SO")" -ge "$MIN_BYTES" ]; then
    echo "seed_physx: already cached ($VERSION, $(stat -c %s "$SO") bytes)"
    exit 0
fi

mkdir -p "$DEST"

# enable_gpu() writes to $HOME, so give it a scratch $HOME and move the result onto NFS.
# The move is atomic within one filesystem, so a reader either sees no file or a complete
# one - never the half-written state that caused the crash.
TMP_HOME="$(mktemp -d)"
trap 'rm -rf "$TMP_HOME"' EXIT
echo "seed_physx: downloading $VERSION via sapien (scratch HOME=$TMP_HOME)"
HOME="$TMP_HOME" "$PY" -c 'import sapien.physx as p; p.enable_gpu()'

SRC="$TMP_HOME/.sapien/physx/$VERSION/libPhysXGpu_64.so"
if [ ! -s "$SRC" ]; then
    echo "seed_physx: sapien did not produce $SRC" >&2
    exit 1
fi
cp "$SRC" "$DEST/.libPhysXGpu_64.so.partial"
mv "$DEST/.libPhysXGpu_64.so.partial" "$SO"
echo "seed_physx: cached $SO ($(stat -c %s "$SO") bytes)"
