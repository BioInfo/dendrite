"""Benchmark records: a state of ~N tokens of real prose (kev's own docs, public, Apache-2.0) and Q questions cycled
from a fixed pool of yes/no, multiple-choice and rating questions. Deterministic for a given (tokenizer, N, Q)."""
from pathlib import Path

import os

import kev

# a source checkout of kev (pip install -e, or KEV_DOCS=<checkout>/docs): an installed wheel carries no docs
KEV_DOCS = Path(os.environ.get("KEV_DOCS") or Path(kev.__file__).resolve().parents[1] / "docs")

POOL = [
    ("Does the document describe a training procedure?", ["yes", "no"]),
    ("Which kind of document is this?", ["model card", "research notes", "release notes", "tutorial"]),
    ("How technical is the writing?", ["introductory", "intermediate", "expert"]),
    ("Does the document report a benchmark number?", ["yes", "no"]),
    ("Which hardware is mentioned most?", ["H100", "L40S", "Apple Silicon", "none"]),
    ("Is a license named in the document?", ["yes", "no"]),
    ("What is the main subject?", ["model architecture", "evaluation", "serving", "data"]),
    ("Does the text mention calibration?", ["yes", "no"]),
    ("How confident is the author about the results?", ["tentative", "moderate", "confident"]),
    ("Is there a stated limitation or caveat?", ["yes", "no"]),
    ("Which audience is the document written for?", ["end users", "researchers", "contributors"]),
    ("Does the document compare against Jev?", ["yes", "no"]),
    ("Rate how well organised the document is.", ["1", "2", "3", "4", "5"]),
    ("Is a dataset named?", ["yes", "no"]),
    ("Which programming language appears in examples?", ["Python", "Rust", "JavaScript", "none"]),
    ("Does the document describe a failure that was fixed?", ["yes", "no"]),
]


def corpus():
    return "\n\n".join(p.read_text() for p in sorted(KEV_DOCS.rglob("*.md")))


def make_record(tok, state_tokens, questions):
    ids = tok(corpus(), add_special_tokens=False)["input_ids"]
    if len(ids) < state_tokens: ids = (ids * (state_tokens // len(ids) + 1))
    state = tok.decode(ids[: state_tokens - 1])
    qs = [{"instr": POOL[i % len(POOL)][0], "options": POOL[i % len(POOL)][1], "label": 0} for i in range(questions)]
    return {"state": state, "questions": qs}
