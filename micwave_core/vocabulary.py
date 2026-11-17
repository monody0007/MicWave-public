"""Speaker vocabulary extensions on top of the built-in normalization and prompt.

The public build has none. A private build ships `private_vocab.py`, whose tables are used as-is:

- `STREAMING_SUBSTITUTIONS`: `(compiled pattern, replacement string)`, applied to every streamed delta.
- `FINAL_RULES`: `(rule id, compiled pattern, match -> replacement)`, final-only, after the built-in rules.
- `PROMPT_TERMS`: canonical spellings added to the transcription prompt's term list.
- `PROMPT_CAVEATS`: clauses `不要把…` appended to the prompt's do-not-rewrite sentence.
"""
from __future__ import annotations

try:
    from .private_vocab import FINAL_RULES, PROMPT_CAVEATS, PROMPT_TERMS, STREAMING_SUBSTITUTIONS
except ModuleNotFoundError as exc:
    if exc.name != f"{__package__}.private_vocab":  # a broken private module must fail loudly
        raise
    STREAMING_SUBSTITUTIONS = ()
    FINAL_RULES = ()
    PROMPT_TERMS = ()
    PROMPT_CAVEATS = ()
