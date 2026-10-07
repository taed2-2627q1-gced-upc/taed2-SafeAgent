from collections import Counter, defaultdict
import re

from taed2_safeagent.data.common import canonical_json, digest
from taed2_safeagent.data.inputs import informative_context


class UnionFind:
    """Join related examples."""

    def __init__(self, size):
        self.parents = list(range(size))

    def find(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def join(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parents[right] = left


def command_fingerprint(command, settings):
    markers = "|".join(re.escape(marker) for marker in settings["comment_markers"])
    suffix = re.compile(r"\s+(?:#\s*|&\s*rem\s+)(?:" + markers + r")\s*$", re.IGNORECASE)
    text = re.sub(r"\s+", " ", command.strip())
    while True:
        stripped = suffix.sub("", text)
        if stripped == text:
            break
        text = stripped
    return re.sub(r"\d+", "<num>", text.casefold())


def character_grams(text, size):
    if len(text) < size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def near_pairs(texts, settings):
    grams = {text: character_grams(text, settings["ngram_size"]) for text in texts}
    frequencies = Counter(gram for values in grams.values() for gram in values)
    numerator = settings["jaccard_numerator"]
    denominator = settings["jaccard_denominator"]
    index = defaultdict(list)
    for text in sorted(grams, key=lambda value: (len(grams[value]), value)):
        current = grams[text]
        minimum = (numerator * len(current) + denominator - 1) // denominator
        prefix = sorted(current, key=lambda gram: (frequencies[gram], gram))[
            : len(current) - minimum + 1
        ]
        candidates = set()
        for gram in prefix:
            candidates.update(
                other
                for other in index[gram]
                if denominator * len(grams[other]) >= numerator * len(current)
            )
        for other in sorted(candidates):
            intersection = len(current & grams[other])
            union = len(current) + len(grams[other]) - intersection
            if denominator * intersection >= numerator * union:
                yield other, text
        for gram in prefix:
            index[gram].append(text)


def build_groups(rows, settings):
    joined = UnionFind(len(rows))
    fingerprints, contexts = {}, {}
    for position, row in enumerate(rows):
        text = command_fingerprint(row["command"], settings)
        if text in fingerprints:
            joined.join(position, fingerprints[text])
        else:
            fingerprints[text] = position
        if informative_context(row["context"]):
            context = canonical_json(row["context"])
            if context in contexts:
                joined.join(position, contexts[context])
            else:
                contexts[context] = position
    edges = 0
    for left, right in near_pairs(fingerprints, settings):
        joined.join(fingerprints[left], fingerprints[right])
        edges += 1
    components = defaultdict(list)
    for position, row in enumerate(rows):
        components[joined.find(position)].append(row["id"])
    groups = {}
    for members in components.values():
        members.sort()
        groups[digest("\n".join(members))] = members
    return groups, {
        "fingerprints": len(fingerprints),
        "near_edges": edges,
        "groups": len(groups),
        "largest_group": max(map(len, groups.values()), default=0),
    }
