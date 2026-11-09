"""Advisory checks for incomplete prose, never inferred provider finish reasons."""

import re

# Short French words can legitimately end a sentence; unknown short tokens are
# only suspicious, not proof of truncation (e.g. 'masse no. Une ...').
SHORT_WORDS = set("a à y en on un le la de du vu su eu va ai as es or os an ne ni si se ce ça tu il je me ma ta sa te et ou où au bu lu pu dû né nu mu ri ah eh oh hé ho do ré mi fa ci là us ex ès ô".split())


def check_text_integrity(text: str) -> list[str]:
    warnings = []
    for number, match in enumerate(re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text), 1):
        paragraph = match.group().rstrip()
        position = f"Paragraphe {number}, caractère {match.start() + 1}"
        if paragraph and paragraph[-1] not in '.!?…»"”)':
            warnings.append(f"{position} : fin sans ponctuation de clôture — {paragraph[-100:]!r}")
        for fragment in re.finditer(r"\b([A-Za-zÀ-ÖØ-öø-ÿ]{1,2})\.\s+(?=[A-ZÀ-ÖØ-Þ])", paragraph):
            if fragment.group(1).casefold() not in SHORT_WORDS:
                warnings.append(f"{position} : fragment possiblement coupé {fragment.group(1)!r}, caractère {match.start() + fragment.start() + 1}. Vérifie le raccord.")
    return warnings
