'''
Detector counts for the critic (generate_critic_hf_parquet.py): VLMs miscount, so the number of each object the
caption counts is measured with SAM 3, handed to the VLM in the user turn, and enforced on its reply.

Targets: the caption's counted objects and their numbers. From the `vqa_list` column when the row has one (conceptmix:
["How many grape are in the image?", "five"]), else parsed from the caption ("three pink cookies").

Measurement: SAM 3 concept segmentation, one query per object, all queries of a batch of images in a few forward passes
(the image encoder runs once per image). Instances are cleaned like enrich_captions.py (NMS, group boxes, nested
boxes, tiny boxes). Per object one of:
    exact       n confident instances (score >= --count_high) and no borderline one
    range       n confident and g borderline instances (score in [--count_low, --count_high)): the count is n..n+g
    unmeasured  no confident instance (SAM 3 misses stylised objects: gummy bears, cartoon samurai, and its
                borderline boxes on them undercount: "zero or one" griffin for two), above
                --count_max, or a head noun two targets share ("an enormous parachute and a tiny parachute": SAM 3
                counts every parachute for both queries)

Checked by eye on 20 conceptmix images (2026-10-12): every exact count was right (monkeys 4, airplanes 5, llamas 5,
padlocks 5, life jackets 11, ...); the true count was inside the range in 10 of 11 ranges (skateboards 7..9 -> 9,
cats 4..6 -> 5; delivery vans 1..2 -> 3 was not); "no instance" was wrong as often as right. Grounding DINO was
dropped: it boxes absent objects (wifi routers, vests, hamsters) and overcounts (9 monkeys for 4), and only 28 of 198
objects got a count when both detectors had to agree.

Use (check_counts) -- a count is "decisive" when it shows the caption's number is wrong (exact n != target, or the
target outside the range). Count fixes are written as Add / Remove steps, the form the edit and iterative-generation
data teach (hqedt: "Add two bicycles ...", "Add a second ginger cat ..."; "Change the number of X from A to B" only
appears where the decompose prompt asked for it):
    - decisive and target >= 2: the reply must fix it with "Add k X" (too few) or "Remove k X" (too many), where the
      measured count plus or minus k is the target ("a second", "another", "one more" are k = 1), or with
      "Replace ... with <target> X" (wrong kind of object). Self-reflect counts must match the measurement.
    - any state: a Replace that swaps k X for a different number of X ("Replace the two dartboards with four
      dartboards") is a count fix in disguise and is rejected (replace_changes_count).
    - narrow range ("four or five") containing the target: the VLM decides; a count it states must be in the range and
      Add / Remove steps must end on the target, as above, but nothing is required.
    - not decisive (count right, wider range containing the target, unmeasured): no step may imply a current count of X
      ("Add two more", "Add a third", "Remove two", "in total") and no Self-reflect item may state one; "Add <target> X"
      is still allowed for an object that is missing altogether. Precision over recall: a guessed count teaches the
      critic to guess.
'''

from __future__ import annotations

import json
import re

import torch
from PIL import Image

NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                "sixteen seventeen eighteen nineteen twenty").split()
NUMBERS = {word: value for value, word in enumerate(NUMBER_WORDS)}
_NUMBER = r"(?:" + "|".join(NUMBER_WORDS) + r"|\d+)"

HOW_MANY_RE = re.compile(r"(?i)^how many (.+?) (?:are|is) (?:there )?in the (?:image|picture)\??$")
# Caption fallback: "<two..twenty>" and the words up to the first plural noun, at most four ("three pink cookies",
# "five lambs outside a tent" -> "lambs"); a count of two or more ends on a plural noun.
_WORD_RE = re.compile(r"[A-Za-z0-9][\w'-]*|[^\w\s]")
_SAME_PLURAL = set("sheep fish deer moose bison salmon trout aircraft".split())
# Count claims in a reply: a number up to four words before the noun ("shows seven planets", "four porcelain sleighs"),
# Add / Remove steps with a number ("Add two bicycles", "Add a second cat", "Remove one cow"), "Change the number of X
# from A to B" (not used: count fixes are Add / Remove steps) and steps that imply a total ("to make five in total").
COUNT_STEP_RE = re.compile(r"(?i)\bnumber of (?P<obj>[^,.;]+?)\b(?:from|to)\b")
RELATIVE_RE = re.compile(r"(?i)\b(?:more|additional|extra|another|in total|a total|total of|total count)\b"
                         r"|\b(?:a|an)\s+(?:second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b")
TOTAL_RE = re.compile(r"(?i)\bin total\b|\ba total\b|\btotal of\b|\btotal count\b")
_ORDINALS = set("second third fourth fifth sixth seventh eighth ninth tenth".split())
# A number that counts something other than the whole group: "the two frogs on top", "by adding two frogs".
_NOT_A_TOTAL = {"the", "adding", "removing", "add", "remove", "of", "these", "those", "both"}
_TOKEN_RE = re.compile(r"[\w'-]+")

UNIQUE_OBJECTS = set("sun moon sky earth horizon rainbow".split())
_IRREGULAR = {"people": "person", "men": "man", "women": "woman", "children": "child", "mice": "mouse",
              "geese": "goose", "teeth": "tooth", "feet": "foot", "knives": "knife", "leaves": "leaf",
              "wolves": "wolf", "shelves": "shelf", "loaves": "loaf", "calves": "calf", "scarves": "scarf"}


def to_number(word: str) -> int | None:
    word = word.lower()
    return int(word) if word.isdigit() else NUMBERS.get(word)


def number_word(value: int) -> str:
    return NUMBER_WORDS[value] if 0 <= value < len(NUMBER_WORDS) else str(value)


def singular(word: str) -> str:
    word = word.lower().strip("'\"")
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    for suffix, replacement in (("ies", "y"), ("ches", "ch"), ("shes", "sh"), ("sses", "ss"), ("xes", "x"), ("zes", "z")):
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[:-len(suffix)] + replacement
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    for plural, single in _IRREGULAR.items():  # "firemen", "bread knives"
        if word.endswith(plural):
            return word[:-len(plural)] + single
    return word


def head(noun: str) -> str:
    return singular(noun.split()[-1])


# ------------------------------------------------------------------------------------------------- targets

# Words that end a noun phrase when walking back from its head: determiners, numbers, prepositions, verbs. Comparatives
# are relations to another object, not properties SAM 3 can see ("smaller anteater").
_PHRASE_STOP = set("""a an the some several many few lone single with and or of in on at to by near beside behind above
below under over from into onto than while where that which who is are was were sits sit stands stand rests rest lies
lie hangs hang no without not each both all its their his her this these those around across through along against
between among inside outside atop beneath underneath alongside next nearby there here as taller shorter bigger smaller
larger higher lower longer wider same equal other another""".split())


def noun_phrase(caption: str, noun: str, max_extra: int = 3) -> str:
    """The noun as the caption writes it ("cake" for "cakes" in "a crimson cake") plus up to `max_extra` preceding
    modifiers ("crimson cake" with max_extra >= 1), stopping at a number, determiner, preposition or verb. The noun
    itself when the caption does not contain it (a VQA paraphrase)."""
    tokens = [token.lower() for token in _WORD_RE.findall(caption)]
    words = noun.lower().split()
    head_word = singular(words[-1])
    for j, token in enumerate(tokens):
        if singular(token) != head_word:
            continue
        start = j - len(words) + 1
        if start < 0 or any(singular(tokens[start + i]) != singular(words[i]) for i in range(len(words) - 1)):
            if len(words) > 1:
                continue
            start = j
        extra = 0
        while start > 0 and extra < max_extra:
            before = tokens[start - 1]
            if not before[0].isalpha() or before in _PHRASE_STOP or before in NUMBERS or before.endswith("ing"):
                break
            start -= 1
            extra += 1
        return " ".join(tokens[start:j + 1])
    return noun


def count_targets(row: dict, caption: str, query_modifiers: int = 0) -> list[dict]:
    """[{"noun", "target", "query"}] for the objects the caption counts, from vqa_list when the row has it. "noun" is
    the name the critic's reply is matched against (the VQA question's, usually a bare plural: "cakes"); "query" is
    what SAM 3 is asked for: the noun as the caption writes it ("cake" for "a cake"), with up to `query_modifiers` of
    the caption's modifiers ("crimson cakes").

    Measured 2026-10-18 on the 100 audited exact-count disagreements (right / wrong / unmeasured) and on 1161 targets
    of 400 random rows (measured, i.e. not "unmeasured"):
        VQA noun                81 / 19 / 0    51% measured
        caption-form noun       81 / 17 / 2    60% measured   <- default (query_modifiers=0)
        full phrase (3 mods)    77 / 16 / 7    60% measured
    The caption's number form is what matters: SAM 3 often finds nothing for "farmers" in an image of one farmer and
    finds it for "farmer". Modifiers cost recall ("small kings", "glass gift boxes", "navy blue pitchforks" all went
    unmeasured) and do not buy precision beyond that: most lookalike errors (sheep counted as opossums) are
    cross-category and the modifier does not help, and one of them already used the full phrase and still failed."""
    vqa = row.get("vqa_list")
    targets, seen = [], set()
    if vqa:
        if isinstance(vqa, str):
            vqa = json.loads(vqa)
        for question, answer in vqa:
            match = HOW_MANY_RE.match(question.strip())
            value = to_number(str(answer))
            if match and value is not None and match.group(1).lower() not in seen:
                noun = match.group(1).lower()
                seen.add(noun)
                targets.append({"noun": noun, "target": value, "query": noun_phrase(caption, noun, query_modifiers)})
        return targets
    tokens = [token.lower() for token in _WORD_RE.findall(caption)]
    for k, token in enumerate(tokens):
        if NUMBERS.get(token, 0) < 2:
            continue
        words = tokens[k + 1:k + 5]
        end = next((j for j, word in enumerate(words)
                    if word in _SAME_PLURAL or word in _IRREGULAR or singular(word) != word), None)
        # Stop at punctuation, another number or "of" ("two of its puppies" is not a count of "of its puppies").
        if end is None or any(not word[0].isalpha() or word in NUMBERS or word == "of" for word in words[:end + 1]):
            continue
        noun = " ".join(words[:end + 1])
        if noun not in seen:
            seen.add(noun)
            targets.append({"noun": noun, "target": NUMBERS[token], "query": noun})
    return targets


# ------------------------------------------------------------------------------------------------- measurement

def _expand_rows(value, index: torch.Tensor):
    """Pick rows of a batched model output field (tensor, or tuple of tensors) by `index`."""
    if isinstance(value, torch.Tensor):
        return value.index_select(0, index) if value.dim() else value
    if isinstance(value, (tuple, list)):
        return type(value)(_expand_rows(v, index) for v in value)
    return value


class ObjectCounter:
    """SAM 3 counts for (image, noun) pairs, batched over images and nouns."""

    def __init__(self, args, device: str = "cuda"):
        from transformers import Sam3Model, Sam3Processor

        # The box clean-up rules are shared with enrich_captions.py, where they were tuned.
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reduce_gap_w_ood"))
        from enrich_captions import _area, _contains, clean_boxes

        self._area, self._contains, self._clean_boxes = _area, _contains, clean_boxes
        self.args, self.device = args, device
        self.processor = Sam3Processor.from_pretrained(args.counter_path)
        self.model = Sam3Model.from_pretrained(args.counter_path, dtype=torch.bfloat16).to(device).eval()

    @torch.inference_mode()
    def _instances(self, images: list[Image.Image], pairs: list[tuple[int, str]]) -> list[list[dict]]:
        """Cleaned SAM 3 instances (box, score) per (image index, noun) pair."""
        out: list[list[dict]] = []
        image_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        vision = self.model.get_vision_features(pixel_values=image_inputs["pixel_values"].to(torch.bfloat16))
        sizes = image_inputs["original_sizes"].tolist()
        start, size = 0, self.args.count_batch_size
        while start < len(pairs):
            chunk = pairs[start:start + size]
            try:
                index = torch.tensor([i for i, _ in chunk], device=self.device)
                text_inputs = self.processor(text=[noun for _, noun in chunk], return_tensors="pt").to(self.device)
                embeds = type(vision)(**{key: _expand_rows(value, index) for key, value in vision.items()})
                outputs = self.model(vision_embeds=embeds, input_ids=text_inputs["input_ids"],
                                     attention_mask=text_inputs.get("attention_mask"))
            except torch.OutOfMemoryError:  # the VLM on the same GPU leaves little room
                if size == 1:
                    raise
                torch.cuda.empty_cache()
                size = self.args.count_batch_size = max(1, size // 2)
                continue
            results = self.processor.post_process_instance_segmentation(
                outputs, threshold=self.args.count_low, mask_threshold=0.5, target_sizes=[sizes[i] for i, _ in chunk])
            for result in results:
                boxes = result["boxes"].float().cpu().tolist()
                scores = result["scores"].float().cpu().tolist()
                out.append([{"box": boxes[k], "score": scores[k]} for k in self._clean_boxes(boxes, scores)])
            start += len(chunk)
        return out

    def _evidence(self, found: list[dict], target: dict, shared: bool, image: Image.Image) -> dict:
        noun, value = target["noun"], target["target"]
        evidence = {"noun": noun, "query": target.get("query", noun), "target": value, "state": "unmeasured",
                    "lo": None, "hi": None}
        if shared or head(noun) in UNIQUE_OBJECTS:
            return evidence
        min_area = self.args.count_min_box_frac * image.width * image.height
        strong = [d for d in found if d["score"] >= self.args.count_high]
        # Borderline boxes that are not a duplicate or a part of a confident one, plus confident boxes too small to
        # count (the test prompts count salient objects; tiny ones only say there may be more).
        gray = sum(1 for d in found if d["score"] < self.args.count_high and not any(
            self._contains(s["box"], d["box"], 0.5) or self._contains(d["box"], s["box"], 0.5) for s in strong))
        gray += sum(1 for d in strong if self._area(d["box"]) < min_area)
        lo = sum(1 for d in strong if self._area(d["box"]) >= min_area)
        hi = lo + gray
        # No confident instance: SAM 3 does not recognise the object here (stylised griffins, aliens, dalmatians read
        # as "zero or one" in the 64-row check, each with 2 to 6 in the image), so borderline ones say nothing.
        if lo == 0 or hi > self.args.count_max:
            return evidence
        evidence.update(state="exact" if gray == 0 else "range", lo=lo, hi=hi)
        return evidence

    def __call__(self, images: list[Image.Image], targets: list[list[dict]]) -> list[list[dict]]:
        """Evidence per target per image (see the module docstring). Images go through SAM 3 --count_batch_size at a
        time: its activations grow with the images and queries of a pass, and the VLM leaves it little room."""
        evidence = []
        step = self.args.count_batch_size
        for start in range(0, len(images), step):
            evidence += self._measure(images[start:start + step], targets[start:start + step])
        return evidence

    def _measure(self, images: list[Image.Image], targets: list[list[dict]]) -> list[list[dict]]:
        pairs = [(i, t.get("query", t["noun"])) for i, row in enumerate(targets) for t in row]
        if not pairs:
            return [[] for _ in images]
        found = iter(self._instances(images, pairs))
        evidence = []
        for image, row in zip(images, targets):
            heads = [head(t["noun"]) for t in row]
            evidence.append([self._evidence(next(found), t, heads.count(heads[k]) > 1, image)
                             for k, t in enumerate(row)])
        return evidence


# ------------------------------------------------------------------------------------------------- use

def narrow(e: dict) -> bool:
    """A range of two values ("four or five": one borderline instance). The VLM may pick one; its count is checked."""
    return e["state"] == "range" and e["hi"] - e["lo"] == 1


def decisive(e: dict) -> bool:
    if e["state"] == "exact":
        return e["lo"] != e["target"]
    if e["state"] == "range":
        return not e["lo"] <= e["target"] <= e["hi"]
    return False


def consistent(e: dict, value: int) -> bool:
    return e["state"] in ("exact", "range") and e["lo"] <= value <= e["hi"]


def measured(e: dict) -> str:
    """"exactly four" or "between seven and nine" for a measured count."""
    if e["state"] == "exact":
        return f"exactly {number_word(e['lo'])}"
    if narrow(e):
        return f"{number_word(e['lo'])} or {number_word(e['hi'])}"
    return f"between {number_word(e['lo'])} and {number_word(e['hi'])}"


def evidence_text(evidence: list[dict]) -> str:
    """The user-turn block the VLM reads; empty when nothing is counted."""
    lines = []
    for e in evidence:
        asks = f"the caption asks for {number_word(e['target'])}"
        if e["state"] in ("exact", "range"):
            lines.append(f"- {e['noun']}: {measured(e)} ({asks})")
        else:
            lines.append(f"- {e['noun']}: not measured ({asks})")
    if not lines:
        return ""
    return "Detector counts:\n" + "\n".join(lines)


def _names(text: str, e: dict) -> bool:
    return head(e["noun"]) in {singular(word) for word in _TOKEN_RE.findall(text)}


# Words that end the object of an Add / Remove step: "Add two anteaters | smaller than the steel drum", "Add two ribbons |
# on the dirt road".
_PHRASE_END = set("""than that which who so to and or while with without near next beside behind in on at of from by
under over above below between among around inside outside into onto along across against toward towards for as
then where until positioned placed standing sitting lying resting floating nearby atop beneath underneath alongside
opposite amid""".split())
_QUANTIFIERS = {"a", "an", "the", "another", "more", "additional", "extra", "remaining", "all", "some"} | _ORDINALS


def _is_noun(words: list[str], j: int, e: dict) -> bool:
    """words[j] is e's noun: its singular is the head, and for a compound or modified noun ("spray bottles") the word
    before it is the noun's own modifier or a number / quantifier ("four bottles"), not another one ("pump bottles")."""
    noun_words = e["noun"].lower().split()
    if singular(words[j]) != singular(noun_words[-1]):
        return False
    if len(noun_words) == 1 or j == 0:
        return True
    before = words[j - 1]
    return before in noun_words or to_number(before) is not None or before in _QUANTIFIERS


def _numbers_before(text: str, e: dict) -> list[int]:
    """Numbers that count e's noun in `text`: a number word up to four words before the noun, with no other number
    in between, not a partial count ("the two frogs", "by adding two frogs"), and not "one" (mostly a pronoun)."""
    tokens = [token.lower() for token in _WORD_RE.findall(text)]
    values = []
    for k, token in enumerate(tokens):
        value = to_number(token)
        if value is None or value == 1 or (k and tokens[k - 1] in _NOT_A_TOTAL):
            continue
        for j in range(k + 1, min(k + 5, len(tokens))):
            # "two ornaments inside the chest" counts ornaments, not chests
            if to_number(tokens[j]) is not None or not tokens[j][0].isalpha() or tokens[j] in _PHRASE_END:
                break
            if _is_noun(tokens, j, e):
                values.append(value)
                break
    return values


def count_edit(step: str, e: dict) -> tuple[str, int | None, bool, bool] | None:
    """(op, k, relative, definite) for an Add / Remove step whose object is e's noun, else None.
    op: "add" | "remove". k: how many (None: "Remove the hammers", all of them). relative: the objects join ones already
    there ("more", "another", "a second", "in total"). definite: "Remove the hammer", one specific object."""
    tokens = [token.lower() for token in _WORD_RE.findall(step)]
    if not tokens or tokens[0] not in ("add", "remove"):
        return None
    phrase = []
    for token in tokens[1:]:
        if token in _PHRASE_END or not token[0].isalnum():
            break
        phrase.append(token)
    end = next((j for j in range(len(phrase)) if _is_noun(phrase, j, e)), None)
    if end is None:
        return None
    words = phrase[:end]
    relative = bool(RELATIVE_RE.search(" ".join(words))) or bool(TOTAL_RE.search(step))
    numbers = [to_number(word) for word in words if to_number(word) is not None]
    plural = singular(phrase[end]) != phrase[end]
    if numbers:
        return tokens[0], numbers[0], relative, False
    if words[:1] in (["a"], ["an"], ["another"]) or any(word in _ORDINALS for word in words):
        return tokens[0], 1, relative, False
    if words[:1] == ["the"] and tokens[0] == "remove" and not relative:
        return tokens[0], None if plural else 1, False, True
    if relative:
        return tokens[0], None, True, False  # "Remove the extra teacups", "Add more chairs": a count, but no number
    return None  # "Add chairs", "Remove the hamsters' collars"


def replace_changes_count(step: str, e: dict) -> bool:
    """A Replace step that swaps k of e's noun for a different number of the same noun ("Replace the two dartboards
    with four dartboards", "Replace the three sharks with two hammerhead sharks"): a count fix dodging the Add / Remove
    form. "Replace the bunch of grapes with five grapes" and "Replace the five grapes with a single grape" count the
    noun on one side only and pass."""
    if not step.startswith("Replace"):
        return False
    # Like _numbers_before, but "the two dartboards" counts here: on the replaced side it is the whole set.
    tokens = [token.lower() for token in _WORD_RE.findall(step)]
    values = []
    for k, token in enumerate(tokens):
        value = to_number(token)
        if value is None or value == 1:
            continue
        for j in range(k + 1, min(k + 5, len(tokens))):
            if to_number(tokens[j]) is not None or not tokens[j][0].isalpha() or tokens[j] in _PHRASE_END:
                break
            if _is_noun(tokens, j, e):
                values.append(value)
                break
    return len(values) >= 2 and values[0] != values[-1]


def check_counts(result: dict, evidence: list[dict]) -> str | None:
    """Why the reply's counts disagree with the detector, or None. See the module docstring for the rules."""
    if not evidence:
        return None
    for step in result["correction"]:
        if step.startswith("Change") and COUNT_STEP_RE.search(step):
            return (f'Correction step "{step}" changes a number. Write count fixes as "Add <k> <objects> ..." or '
                    f'"Remove <k> <objects>".')
        for e in evidence:
            if replace_changes_count(step, e):
                return (f'Correction step "{step}" changes how many {e["noun"]} there are inside a replace. Replace '
                        f'the same number ("Replace the three sharks with three hammerhead sharks") and fix the count '
                        f'with "Add <k> {e["noun"]} ..." or "Remove <k> {e["noun"]}".')
    for e in evidence:
        noun, target = e["noun"], e["target"]
        said = [(value, item) for item in result["self_reflect"] for value in _numbers_before(item, e) if value != target]
        edits = [(step, edit) for step in result["correction"] if (edit := count_edit(step, e))]
        totals = [step for step in result["correction"]
                  if TOTAL_RE.search(step) and _names(step, e) and not count_edit(step, e)]
        if totals:
            return f'Correction step "{totals[0]}" implies how many {noun} there are. Say only what to add or remove.'
        if decisive(e) or narrow(e):
            for value, item in said:
                if not consistent(e, value):
                    return (f'The Self-reflect item "{item}" says {number_word(value)} {noun}, but the detector counts '
                            f"{measured(e)}.")
            if edits:
                # Apply the steps in order to every count the detector allows; one must end on the target ("Remove
                # the hammer from his hand" then "Add two hammers on the ground": 1 - 1 + 2 = 2).
                counts = set(range(e["lo"], e["hi"] + 1))
                for step, (op, k, relative, definite) in edits:
                    if k is None and op == "add":
                        return (f'Correction step "{step}" does not say how many {noun} to add: the detector counts '
                                f"{measured(e)}, so add the difference.")
                    if k is None and relative:  # "Remove the extra grapes": down to the caption's number
                        counts = {min(n, target) for n in counts}
                    else:
                        counts = {0 if k is None else n + k if op == "add" else max(n - k, 0) for n in counts}
                if target not in counts and not (len(edits) == 1 and edits[0][1][0] == "add"
                                                 and edits[0][1][1] == target and e["lo"] == 0):
                    verb = "add" if target > e["hi"] else "remove"
                    return (f"The steps on {noun} ({'; '.join(step for step, _ in edits)}) do not end at "
                            f"{number_word(target)}: the detector counts {measured(e)}, so {verb} the difference.")
                continue
            replaced = any(target in _numbers_before(step, e) for step in result["correction"] if step.startswith("Replace"))
            if decisive(e) and target >= 2 and not replaced:
                verb = "Add" if target > e["hi"] else "Remove"
                return (f"The detector counts {measured(e)} {noun} but the caption asks for {number_word(target)}. "
                        f'Report it and fix it with "{verb} <k> {noun} ...", k being the difference.')
        else:
            if e["state"] == "exact":
                why = f"the detector counts exactly {number_word(e['lo'])} {noun}, as the caption asks"
            else:
                why = f"the detector could not verify how many {noun} there are"
            for step, (op, k, relative, definite) in edits:
                if definite:
                    continue  # "Remove the hammer from his hand": one specific object, not a count
                if op == "remove" or relative or k != target:
                    return (f'Correction step "{step}" implies how many {noun} there are, but {why}. Do not add or '
                            f"remove some of them; add all {number_word(target)} only if none is there.")
            if said:
                return (f'The Self-reflect item "{said[0][1]}" states how many {noun} there are, but {why}. Do not '
                        f"state a count for {noun}.")
    return None
