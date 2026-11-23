"""Text normalisation with TensorFlow: captions -> hashtag candidate terms.

Pipeline (all string ops run as TensorFlow graph ops):
  1. standardise    tf.strings.lower + tf.strings.regex_replace (keep a-z only)
  2. tokenise       tf.strings.split  -> RaggedTensor of tokens
  3. vocabulary     tf.keras.layers.TextVectorization.adapt() on TRAINING captions only
  4. normalise      tf.lookup.StaticHashTable: plural -> singular
                    (cars -> car, puppies -> puppy, dishes -> dish). The table is learned from
                    the training vocabulary: a plural is merged only if its singular form is
                    itself a word in the corpus, with an exception list (news, glass, bus ...).
  5. filter         a second StaticHashTable flags stop-words, function words, generic words
                    and visual-detail words; tokens shorter than 3 characters are dropped.

Compared with a linguistic pipeline (POS tagging + lemmatisation), this keeps verb forms as
written (cooking, cook and cooks->cook stay separate) and relies on the stop-list rather than
part-of-speech to drop adjectives and function words.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402

# English function words (articles, pronouns, prepositions, auxiliaries, conjunctions, adverbs)
FUNCTION = set("""
a an the and or but nor so yet for of in on at to from by with without within into onto upon over under
above below between among through during before after behind beside besides near off out up down across
along around about against toward towards via per than then there here where when while whilst why how what
which who whom whose that this these those it its itself he him his himself she her hers herself they them
their theirs themselves we us our ours you your yours i me my mine myself one ones some any all both each
every either neither no not none nothing nobody nowhere somewhere anywhere everywhere also just only even
very too quite really still again ever never always often sometimes usually soon now later already almost
much many more most less least few several such own same other others another else enough rather
is are was were be been being am do does did doing done have has had having will would shall should can
could may might must ought get gets got getting gotten
as if because since unless until whether though although once like unlike
yes ok okay oh hey well
""".split())

# Generic words that make useless hashtags (#man, #video, #talking ...).
GENERIC = set("""
man woman person people guy girl boy lady men women kid kids someone somebody everyone everybody
video clip thing something stuff way front side top lot group bunch part piece time day kind sort type
scene shot footage view image picture camera background screen
talk talks talking talked speak speaks speaking say says saying said tell tells telling told
show shows showing shown showed play plays played make makes making made go goes going went gone
look looks looking looked stand stands standing stood sit sits sitting sat walk walks walking walked
come comes coming came take takes taking took give gives giving gave put puts putting use uses using used
try tries trying tried see sees seeing saw seen watch watches watching watched hold holds holding held
run runs running ran move moves moving moved start starts starting started turn turns turning turned
keep keeps keeping kept let lets seem seems want wants wanting need needs know knows think thinks
doe does
""".split())

# Visual-detail words annotators mention often but nobody would hashtag (#shirt, #chair ...).
LOW_VALUE = set("""
shirt wearing wear wears jacket glass glasses hat cap suit tie coat blue pink orange blonde color colors
line hole item eye eyes face hand hands head leg legs body chair table desk door floor wall window seat
room ground bag box bottle cup stick letter word words text laying lying placing pointing pulling
removing covering displaying describing asking question opening rolling falling feature ping pong bill
sings playing work working living motion night light
""".split())

# Adjectives / adverbs / numbers that a part-of-speech filter would have removed.
DESCRIPTIVE = set("""
young old new big small little large tiny huge long short tall high low good bad great nice best better
first second third last next different various several certain main real full empty open closed
black white red green yellow purple brown gray grey dark bright colorful
two three four five six ten hundred thousand
fast slow quick quickly slowly together alone away back home inside outside around forward
funny happy sad beautiful pretty cute hot cold angry wooden older female male late ready live
asian indian spanish
""".split())

# Found by inspecting the first TensorFlow vocabulary build (no POS filter to remove them).
REVIEWED = set("""
add adding bra care david display displayed describe fall human leave place potatoe set sign sim world
""".split())

STOP = FUNCTION | GENERIC | LOW_VALUE | DESCRIPTIVE | REVIEWED

# words ending in "s" that are NOT plurals of another corpus word
NOT_PLURAL = set("""
news glass bus gas class dress grass chess boss cross mess kiss press process business actress princess
lens series species physics mathematics gymnastics athletics olympics politics economics graphics
clothes jeans pants shorts scissors sunglasses glasses christmas texas paris james chris thomas
tennis canvas atlas campus virus bonus circus status focus chorus cactus octopus
""".split())


def standardize(texts) -> tf.Tensor:
    x = tf.strings.lower(tf.convert_to_tensor(texts))
    return tf.strings.regex_replace(x, r"[^a-z]+", " ")


def tokenize(texts) -> tf.RaggedTensor:
    return tf.strings.split(tf.strings.strip(standardize(texts)))


def learn_vocabulary(captions: list[str]) -> list[str]:
    """Corpus vocabulary via Keras TextVectorization (fit on training captions only)."""
    tv = tf.keras.layers.TextVectorization(standardize=standardize, split="whitespace", output_mode="int")
    tv.adapt(tf.data.Dataset.from_tensor_slices(captions).batch(4096))
    return [w for w in tv.get_vocabulary() if w and w != "[UNK]"]


def learn_term_map(vocabulary: list[str]) -> dict[str, str]:
    """plural -> singular, only when the singular is itself a corpus word."""
    vocab = set(vocabulary)
    out = {}
    for w in vocabulary:
        if len(w) < 4 or not w.endswith("s") or w in NOT_PLURAL or w.endswith(("ss", "us", "is", "ics")):
            continue
        for cand in ([w[:-3] + "y"] if w.endswith("ies") else []) + \
                    ([w[:-2]] if w.endswith(("ses", "xes", "zes", "ches", "shes")) else []) + [w[:-1]]:
            if cand in vocab and cand != w:
                out[w] = cand
                break
    return out


class TermExtractor:
    """caption -> set of hashtag candidate terms, implemented with TensorFlow lookup tables."""

    def __init__(self, term_map: dict[str, str]):
        self.term_map = dict(term_map)
        keys = list(self.term_map) or ["__none__"]
        vals = [self.term_map.get(k, k) for k in keys]
        self.map_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(vals)), default_value="")
        stop = sorted(STOP)
        self.stop_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(stop), tf.ones(len(stop), tf.int32)), default_value=0)

    def terms(self, captions: list[str], batch: int = 50000) -> list[set[str]]:
        out: list[set[str]] = []
        for b in range(0, len(captions), batch):
            tok = tokenize(captions[b:b + batch])
            flat = tok.flat_values
            mapped = self.map_table.lookup(flat)
            flat = tf.where(mapped == "", flat, mapped)                         # plural -> singular
            keep = tf.logical_and(self.stop_table.lookup(flat) == 0, tf.strings.length(flat) >= 3)
            kept = tf.ragged.boolean_mask(tf.RaggedTensor.from_row_splits(flat, tok.row_splits),
                                          tf.RaggedTensor.from_row_splits(keep, tok.row_splits))
            out.extend({t.decode() for t in row} for row in kept.to_list())
        return out
