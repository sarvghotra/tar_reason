"""Oracle critique: the SFT-style reflection that names exactly the VQA
questions a draft failed, built from the judge's per-question probabilities.

Used by the critique-sensitivity probe (eval/analyze/critique_sensitivity.py)
and by GRPO training with ``--critique_source oracle`` (train_grpo.py), where
the refiner is trained to execute a critique that is always correct and
complete, with the critic held fixed.
"""

import re

# The SFT data's exact no-issue reflection.
NO_ISSUE_CRITIQUE = "no issues, it matches the prompt.\nCorrection: looks good."
SKILL_ORDER = ("object", "count", "attribute", "position", "verb")

COUNT_RE = re.compile(r"^How many (.+) are in the image\?$")
OBJECT_RE = re.compile(r"^Are there any (.+) in the image\?$")
BE_RE = re.compile(r"^(Is|Are) the (.+)\?$")


# Plurals of nouns ending in "-ie", which the "-ies" -> "-y" rule would mangle.
IE_PLURALS = {"cookies", "movies", "pies", "ties", "zombies", "brownies", "smoothies"}


def singularize(noun):
    words = noun.split()
    last = words[-1]
    if last in IE_PLURALS:
        last = last[:-1]
    elif last.endswith("ies") and len(last) > 4:
        last = last[:-3] + "y"
    elif re.search(r"(ss|ch|sh|x)es$", last):
        last = last[:-2]
    elif last.endswith("s") and not last.endswith(("ss", "us")):
        last = last[:-1]
    return " ".join(words[:-1] + [last])


def prompt_nouns(vqa_list):
    """Plural object nouns named by the count / object questions, and the
    count each should have (None when the prompt has no count question)."""
    counts = {}
    for question, answer in vqa_list:
        m = COUNT_RE.match(question)
        if m:
            counts[m.group(1)] = answer
        m = OBJECT_RE.match(question)
        if m:
            counts.setdefault(m.group(1), None)
    return counts


def with_count(count, plural):
    return f"one {singularize(plural)}" if count == "one" else f"{count} {plural}"


def split_subject(rest, nouns):
    """'croissant to the left of the dog' -> ('croissant', 'to the left of the dog')."""
    candidates = set()
    for plural in nouns:
        candidates.update({plural, singularize(plural)})
    for noun in sorted(candidates, key=len, reverse=True):
        if rest.startswith(noun + " "):
            return noun, rest[len(noun) + 1:]
    head, _, tail = rest.partition(" ")
    return head, tail


def atom_critique(question, answer, skill, nouns):
    """(issue, fix) sentences for one failed question, in the SFT critique style."""
    m = COUNT_RE.match(question)
    if m:
        plural = m.group(1)
        be = "is" if answer == "one" else "are"
        return (f"the image does not show exactly {with_count(answer, plural)}",
                f"make sure there {be} exactly {with_count(answer, plural)}")
    m = OBJECT_RE.match(question)
    if m:
        plural = m.group(1)
        count = nouns.get(plural)
        return (f"there are no {plural} in the image",
                f"add {with_count(count, plural) if count else plural}")
    m = BE_RE.match(question)
    if m:
        verb = m.group(1).lower()
        subject, predicate = split_subject(m.group(2), nouns)
        if predicate:
            issue = f"the {subject} {verb} not {predicate}"
            if skill == "position":
                pronoun = "it is" if verb == "is" else "they are"
                return issue, f"move the {subject} so that {pronoun} {predicate}"
            if skill == "verb":
                return issue, f"show the {subject} {predicate}"
            return issue, f"make the {subject} {predicate}"
    return (f"the image does not satisfy: {question}",
            f"change the image so that the answer to \"{question}\" is {answer}")


def oracle_critique(vqa_list, skills, per_question, threshold):
    """Critique naming exactly the draft's failed questions (SFT format)."""
    nouns = prompt_nouns(vqa_list)
    failed = [(q, a, s) for (q, a), s, p in zip(vqa_list, skills, per_question)
              if p < threshold]
    if not failed:
        return NO_ISSUE_CRITIQUE
    # A missing object also fails its count question; say it once, as "add N".
    missing = {OBJECT_RE.match(q).group(1) for q, _, _ in failed if OBJECT_RE.match(q)}
    failed = [f for f in failed
              if not (COUNT_RE.match(f[0]) and COUNT_RE.match(f[0]).group(1) in missing)]
    rank = {s: i for i, s in enumerate(SKILL_ORDER)}
    failed.sort(key=lambda f: rank.get(f[2], len(rank)))
    parts = [atom_critique(q, a, s, nouns) for q, a, s in failed]
    return (", and ".join(p[0] for p in parts) + "\nCorrection: "
            + ", and ".join(p[1] for p in parts))
