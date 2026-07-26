"""Fetch and validate a PaliGemma tokenizer from a non-gated mirror.

google/paligemma-3b-pt-224 is gated and we have no HF token, but the tokenizer is
just the Big Vision SentencePiece model. Mirrors carry byte-identical copies; we
verify that by checksumming tokenizer.model against the canonical Big Vision file
served from storage.googleapis.com (public, ungated).

Result is written to assets/paligemma_tokenizer/ and referenced from the pi05-mem
processor via `tokenizer_name`.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "paligemma_tokenizer"
CANONICAL_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"

MIRRORS = [
    "leo009/paligemma-3b-pt-224",
    "Shakalaka/paligemma-3b-pt-224",
    "moaazelmarakby/paligemma-3b-pt-224",
]
PATTERNS = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json", "preprocessor_config.json"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_digest() -> str:
    tmp = ROOT / ".cache" / "paligemma_tokenizer.model"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.exists():
        urllib.request.urlretrieve(CANONICAL_URL, tmp)  # noqa: S310 - fixed https URL
    digest = sha256(tmp)
    print(f"canonical big_vision tokenizer.model sha256={digest}")
    return digest


def main() -> int:
    want = canonical_digest()
    for repo in MIRRORS:
        try:
            path = Path(snapshot_download(repo, allow_patterns=PATTERNS))
        except Exception as exc:  # noqa: BLE001 - mirrors are best-effort
            print(f"{repo}: download failed ({type(exc).__name__}: {exc})")
            continue
        model_file = path / "tokenizer.model"
        if not model_file.exists():
            print(f"{repo}: no tokenizer.model")
            continue
        got = sha256(model_file)
        if got != want:
            print(f"{repo}: tokenizer.model MISMATCH ({got})")
            continue
        print(f"{repo}: tokenizer.model matches canonical")
        if OUT.exists():
            shutil.rmtree(OUT)
        shutil.copytree(path, OUT, symlinks=False)
        break
    else:
        print("no mirror matched the canonical tokenizer")
        return 1

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(OUT))
    print("tokenizer class:", type(tok).__name__)
    print("vocab_size:", tok.vocab_size, "len:", len(tok))
    ids = tok("Task: pick the red cube, State: 12 34;\nAction: ")["input_ids"]
    print("sample ids:", ids[:12], "...")
    assert tok.vocab_size == 257152, f"unexpected vocab_size {tok.vocab_size}"
    assert "<image>" in tok.get_vocab(), "missing <image> token"
    print("TOKENIZER_OK", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
