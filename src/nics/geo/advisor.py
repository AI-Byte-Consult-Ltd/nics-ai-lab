"""Turn class probabilities into an honest answer for a person (ru / en).

What it always does:
  * names the top guesses with their (calibrated) probabilities;
  * says how sure it is in words, and refuses to name anything below ABSTAIN;
  * lists simple checks that tell the top guesses apart (streak, magnet, acid, knife);
  * says what the find MAY indicate, only when the top guess is reasonably sure;
  * ends with the same plain limit: a photo is a hint, an assay is proof.

What it never does: claim a deposit, give a value in money, or speak about oil and gas
from a rock photo. The texts come from taxonomy.py (rules written by people), the model
only supplies the probabilities.
"""
from __future__ import annotations

from . import taxonomy

ABSTAIN = 0.25
SURE = 0.70
MAYBE = 0.40

_T = {
    "ru": {
        "sure": "Скорее всего это {name} ({p} %).",
        "maybe": "Возможно, это {name} ({p} %).",
        "unsure": "Не уверен. Больше всего похоже на {name} ({p} %).",
        "abstain": "Не могу уверенно определить камень по этому фото.",
        "retake": "Сфотографируйте ближе, при дневном свете, на однотонном фоне, свежий скол если есть.",
        "alts": "Другие варианты: {alts}.",
        "check": "Как проверить:",
        "means": "Что это может значить:",
        "caution": "Осторожно:",
        "limit": "Это оценка по фото, а не анализ. Состав и наличие металлов подтверждает только проба в лаборатории.",
    },
    "en": {
        "sure": "Most likely {name} ({p} %).",
        "maybe": "Possibly {name} ({p} %).",
        "unsure": "Not sure. It looks most like {name} ({p} %).",
        "abstain": "I cannot identify this stone with confidence from this photo.",
        "retake": "Take a closer photo in daylight, on a plain background, of a fresh break if possible.",
        "alts": "Other options: {alts}.",
        "check": "How to check:",
        "means": "What it may mean:",
        "caution": "Caution:",
        "limit": "This is an estimate from a photo, not an analysis. Only a lab assay confirms composition and metals.",
    },
}


def answer(probs: dict[str, float], lang: str = "ru", top_k: int = 3) -> dict:
    lang = lang if lang in _T else "en"
    t = _T[lang]
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])[:top_k]
    top_key, top_p = ranked[0]
    top = taxonomy.BY_KEY[top_key]

    def nm(k: str) -> str:
        c = taxonomy.BY_KEY[k]
        return c.ru if lang == "ru" else c.en

    def pct(p: float) -> int:
        return int(round(p * 100))

    lines: list[str] = []
    if top_p < ABSTAIN:
        lines += [t["abstain"], t["retake"]]
        level = "abstain"
    else:
        level = "sure" if top_p >= SURE else "maybe" if top_p >= MAYBE else "unsure"
        lines.append(t[level].format(name=nm(top_key).lower() if lang == "ru" else nm(top_key), p=pct(top_p)))
    alts = [f"{nm(k).lower() if lang == 'ru' else nm(k)} ({pct(p)} %)" for k, p in ranked[1:] if p >= 0.05]
    if alts:
        lines.append(t["alts"].format(alts=", ".join(alts)))

    tests: list[str] = []
    for k, p in ranked[:2]:
        if p >= 0.10:
            for tk in taxonomy.BY_KEY[k].tests:
                tt = taxonomy.TESTS[tk][lang]
                txt = f"{nm(k)}: {tt[0].lower()}{tt[1:]}"
                if txt not in tests:
                    tests.append(txt)
    if tests:
        lines.append(t["check"])
        lines += [f"• {x}" for x in tests]

    means: list[str] = []
    if top_p >= MAYBE:
        means = [taxonomy.INDICATORS[k][lang] for k in top.indicates]
        if means:
            lines.append(t["means"])
            lines += [f"• {x}" for x in means]
    cautions = sorted({ck for k, p in ranked[:2] if p >= 0.10 for ck in taxonomy.BY_KEY[k].caution})
    if cautions:
        lines.append(t["caution"])
        lines += [f"• {taxonomy.CAUTIONS[ck][lang]}" for ck in cautions]
    lines.append(t["limit"])
    return {
        "level": level,
        "top": [{"class": k, "name": nm(k), "group": taxonomy.BY_KEY[k].group, "probability": round(p, 4)} for k, p in ranked],
        "tests": tests, "may_indicate": means, "cautions": [taxonomy.CAUTIONS[c][lang] for c in cautions],
        "text": "\n".join(lines),
    }
