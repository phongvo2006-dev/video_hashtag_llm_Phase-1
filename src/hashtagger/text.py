"""Text normalisation: tokenise, POS-tag and lemmatise captions into hashtag candidates.

Nouns are lemmatised to singular form (``cars`` -> ``car``). Verbs are mapped
to their gerund form (``sings`` / ``sang`` -> ``singing``), the form people
actually use as hashtags (#cooking, #dancing). The gerund form for each verb
is learned from the corpus (most frequent VBG surface form per lemma).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

import nltk
from nltk.stem import WordNetLemmatizer

_TOKEN_RE = re.compile(r"[a-z]+")

# Generic words that make useless hashtags (#man, #video, #talking ...).
GENERIC = set("""
man woman person people guy girl boy lady men women kid someone somebody everyone
video clip thing something stuff other another way front side top lot group bunch
part piece time day kind sort type scene shot footage view image picture camera
talk speak say tell show play do make go get look stand sit walk come take give put
use try see watch hold run move start turn keep let seem want need know think
background screen while about him her his them their they its this that
being have has had are was were been doe
""".split())

# Visual-detail words annotators mention often but nobody would hashtag
# (#shirt, #chair, #wearing). Found by inspecting the first vocabulary build.
LOW_VALUE = set("""
shirt wearing jacket glass hat cap suit tie coat blue pink orange blonde color line
hole item eye face hand head leg body chair table desk door floor wall window seat
room ground bag box bottle cup stick letter word text laying lying placing pointing
pulling removing covering displaying describing asking question opening rolling
falling feature ping pong bill sings playing work working living motion night light
""".split())
GENERIC |= LOW_VALUE

_lemmatizer = WordNetLemmatizer()


def ensure_nltk() -> None:
    """Download the small NLTK resources we need (idempotent)."""
    for pkg, path in [
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def tag_sents(token_lists: list[list[str]]) -> list[list[tuple[str, str]]]:
    return nltk.pos_tag_sents(token_lists)


def learn_gerunds(tagged: Iterable[list[tuple[str, str]]]) -> dict[str, str]:
    """verb lemma -> most frequent VBG surface form (sing -> singing)."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for sent in tagged:
        for w, t in sent:
            if t == "VBG" and len(w) > 4:
                counts[_lemmatizer.lemmatize(w, "v")][w] += 1
    return {lem: c.most_common(1)[0][0] for lem, c in counts.items()}


def candidate_terms(sent: list[tuple[str, str]], gerunds: dict[str, str]) -> set[str]:
    """Hashtag candidate terms in one tagged caption."""
    out: set[str] = set()
    for w, t in sent:
        if len(w) < 3:
            continue
        if t.startswith("NN"):
            lem = _lemmatizer.lemmatize(w, "n")
        elif t.startswith("VB"):
            lem_v = _lemmatizer.lemmatize(w, "v")
            if lem_v in GENERIC:
                continue
            lem = gerunds.get(lem_v)
            if lem is None:
                continue
        else:
            continue
        if lem in GENERIC or len(lem) < 3:
            continue
        out.add(lem)
    return out
