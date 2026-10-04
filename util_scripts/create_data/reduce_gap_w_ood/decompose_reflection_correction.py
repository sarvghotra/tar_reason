'''
The edit (correction) skills in the iterative image generation datasets and the OOD test sets have a gap. So, as a mitigation
this script decomposes the Self-reflection and Correction in atomic skills / steps in numbered lists so that these
can be composed as much as possible to carry out the edit skills needed in the OOD test sets.

In order to prompt the llm model for the pre-processing, get some examples from the datasets to understand what the
self-reflection and correction looks like and how composing correction steps can help to improve the iteration-1 (refined) images
results in OOD test sets:
1. Geneval2: results/GenEval2/iter_adhoc_slf_ref_edit_t20_16K_greedy_slf_ref_draft_S0_seed13_512px_repeat1
2. TIIF: results/tiif-bench-testmini_eval/iter_adhoc_slf_ref_edit_t20_16K_greedy_slf_ref_draft_S0_512px_repeat1
3. Genai-bench: results/genai-bench-800/iter_adhoc_slf_ref_edit_t20_16K_greedy_slf_ref_draft_S0_512px_repeat1

Write the output in the same data by renaming the current "conversations" columns to "v1-conversations" and the new processed as
"conversations"

Input:
    conversations:
        human: ...
        gpt: <image>\nSelf-reflection:...\nCorrection: ...\n<image>

Do not use vllm yet. Would like to avoid vllm startup overhead to iterate quickly.

--------------------------------------------------------------------------------------------------------------------
Implementation notes
--------------------------------------------------------------------------------------------------------------------

Why decompose. In the OOD result dirs above, the SFT model's critique usually names the right failure, but the refined
image fixes it only 14-20% of the time. The corrections it writes bundle several operations into one run-on sentence
("Add two more bicycles to make a total of seven, and change the m..."), while the training corrections that it learned
to execute are the same kind of bundle ("replace X, remove Y and Z, add W"). Rewriting every training critique and
correction as a numbered list of editing goals gives the model one goal per line, so an OOD fix such as "seven
bicycles, not five, to the right of the lion" can be composed from goals it has seen in isolation.

A goal is one editing action (add, remove, replace/swap, move, change in place) applied to one target object. A new
step starts only when the target or the action changes; everything the source does to the same target with the same
action stays one step, however many attributes it touches ("reduce the saturation and increase the contrast of the
image" is one change step, "remove the lamp and the rug" is two remove steps). Steps are not sentences: the rewrite
neither splits one goal into sub-steps nor keeps two goals in one sentence. Preservation constraints ("keep the background
unchanged", "while maintaining the pose") are not goals and are dropped; coverage is measured against the Correction with
those clauses removed (edit_text).

Row formats handled (the marker spelling found in the row is kept in the output):

    hqedt   gpt:   <image>\nSelf-reflect: <critique>\nCorrection: <instruction>\n<image>
      ->    gpt:   <image>\nSelf-reflect:\n1. ...\n2. ...\nCorrection:\n1. ...\n2. ...\n<image>

    UnicEdit human: <image>\nCorrection: <instruction>          gpt: <image>
      ->    human: <image>\nCorrection:\n1. ...\n2. ...           gpt: <image>

Pipeline (text and images are handled in separate passes, so the LLM never waits on 6 GB image shards):

    1. Read only the conversation column of every shard and build one LLM job per row.
    2. Generate with Qwen3.6-27B through Hugging Face Transformers (no vLLM), non-thinking mode, the model card's
       instruct sampling (temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5). Replies are JSON, validated
       (every step is a non-empty single line, no <image> token, most content words of the original survive), and a
       failed reply is retried with the rejected reply and the reason it failed added to the conversation. Results go to a JSONL cache keyed by a hash of prompt version + row text, so a re-run
       with the same prompt skips finished rows and a prompt change regenerates everything.
    3. Rewrite each shard row group by row group: the source "conversations" column is kept as "v1-conversations" and
       the decomposed conversations are written as "conversations". Row-group sizes and the compression codec are
       preserved, and the shard is replaced atomically. If a shard already has "v1-conversations" (a previous run),
       that column is the source and only "conversations" is regenerated. Rows whose reply never validated keep their
       original text in "conversations" and are counted in the log.

Multi-GPU: launch one process per GPU with --rank / --world_size (see decompose_reflection_correction.sh). Each rank
generates rows i with i % world_size == rank, writes a done-marker, waits for all markers, then rewrites shards
j with j % world_size == rank. Rows with identical text are generated once. For runs that span several jobs, --stage
generate fills the cache without the barrier and a later --stage rewrite (CPU only) writes the shards.

Quick iteration: --dry_run --limit 24 prints before/after for 24 rows per input dir and writes nothing.

Text fixes from the hqedt / hqedt2 quality check (2026-10-02, 64 sampled rows per set plus regexes over all 176k rows):

    - Junk rows are dropped from the output shards (--keep_junk keeps them unchanged instead) and never sent to the
      LLM: template leftovers ("PLACEHOLDER_SELF_REFLECT", "EDIT_OPERATION", "</self_reflect>", ~220 rows), "no edit
      required / the images are identical" replies (~30), and HQ-Edit diptych critiques ("the right panel", "both
      images", "the initial description", ~1%) that describe a two-panel source image rather than the draft.
    - Captions written as an edit delta ("now", "replacing the previous ...", "X is absent", "has been removed",
      "slightly reduced saturation", "instead of ..."; 6-9% of rows) leak the draft into the T2I prompt. Flagged
      captions are rewritten by the same LLM call as a standalone description of the target image. A rewrite that
      still reads as a delta or adds content is retried; after the retries the original caption is kept and the
      decomposed steps are still used.
    - Corrections that narrate a finished edit ("The second image has been altered ...", "The operation to edit the
      first image ...", ~13%) and Self-reflects that leak the edit pair ("is replaced by", "should have been
      removed") are rejected by validation, so their steps are retried until they are imperative / draft-only.
'''

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

# Model loading, the presence-penalty logits processor and the no-think chat-template probe are shared with the
# caption-error script, which already solved Qwen3.6 + Transformers 5.x loading without vLLM.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "add_errors_in_caption"))
from add_errors_in_caption_hf import (  # noqa: E402
    _resolve_no_think_kwargs,
    generate,
    load_model,
)

logger = logging.getLogger("decompose")

input_dirs = [
    "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/gpt-edit-simpler_hqedt_slfref_tar/small_sample",
    "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/UnicEdit-tar-format/small_sample"
]

# Refer the model card https://huggingface.co/Qwen/Qwen3.6-27B for input preparation and sampling parameters
llm_model_path = "/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/Qwen3.6-27B"

SOURCE_COLUMN = "conversations"
V1_COLUMN = "v1-conversations"

# Marker spellings seen in the edit shards ("Self-reflect:" in the gpt-edit shards, "Self-correction:" in HumanEdit).
TURN_RE = re.compile(
    r"(?s)^(?P<pre>.*?)"
    r"(?:(?P<sr_marker>Self-reflect(?:ion)?\s*:)\s*(?P<sr>.*?)\s*)?"
    r"(?P<corr_marker>(?:Self-)?Correction\s*:)\s*(?P<corr>.*?)"
    r"(?P<post>\s*<image>\s*)?$"
)
IMAGE_TOKEN = "<image>"

# Rows the rewrite cannot repair; checked on the Self-reflect + Correction text.
JUNK_PATTERNS = {
    "placeholder": re.compile(r"PLACEHOLDER|EDIT_OPERATION|</?[a-z]+_[a-z_]+>"),
    "no_edit": re.compile(
        r"(?i)\bno edits? (?:operation )?(?:is |are |was )?(?:required|needed|performed)|\bno changes? (?:is |are )?(?:required|needed)"
        r"|\bimages are identical\b|\bno image content\b|\bmatches the description perfectly\b"
    ),
    "diptych": re.compile(r"(?i)\b(?:diptych|(?:left|right|first|second) panel|both (?:images|panels)|initial description)\b"),
}
# Edit-delta wording in a caption: flags it for a rewrite. CAPTION_DELTA_STRICT must not survive the rewrite.
CAPTION_DELTA_STRICT = re.compile(
    r"(?i)\b(?:now|no longer|replacing|previous(?:ly)?|(?:has|have) been|(?:is|are) absent|compared to|than before"
    r"|than (?:in )?the (?:original|first|previous)|removed|replaced|instead of|rather than"
    r"|slightly (?:more|less|reduced|increased|darker|brighter|lighter)"
    r"|(?:increased|reduced|decreased|enhanced) (?:colou?r )?(?:saturation|contrast|brightness|vibranc\w*|sharpness|exposure))\b"
)
# "more X than" is often plain description ("taking up more space than a cat would"): ask for a rewrite, accept it unchanged.
CAPTION_DELTA = re.compile(CAPTION_DELTA_STRICT.pattern + r"|(?i:\bmore \w+ than\b)")
# Edit-pair framing that no output step may contain (narrated edits in the Correction, target leaks in the Self-reflect).
# A second action joined into a step ("change the time of day to night and add a starry sky"). "by adding/removing k" in a
# count step and "with mechanical parts replacing ..." are content, not a second action, and do not match.
SECOND_ACTION = re.compile(
    r"(?i)(?:,|\band)\s+(?:then\s+)?(add(?:ing)?|remov(?:e|ing)|replac(?:e|ing)|delet(?:e|ing)|insert(?:ing)?|chang(?:e|ing)|adjust(?:ing)?|alter(?:ing)?|mov(?:e|ing))\b"
)
# Verbs that name the same action: "change the dress to purple and adjust the lighting" is not rejected (the step text
# alone cannot tell whether the targets differ), "replace the background and adjust the lighting" is.
_ACTION = {"add": "add", "remo": "remove", "dele": "remove", "repl": "replace", "inse": "add", "chan": "change",
           "adju": "change", "alte": "change", "make": "change", "turn": "change", "move": "move", "movi": "move"}


def has_second_action(step: str) -> bool:
    """True when a step joins a second, different action ("add a calendar and change the plates")."""
    first = step.split(maxsplit=1)[0].lower()[:4] if step.split() else ""
    first = _ACTION.get(first, first)
    return any(_ACTION.get(match.group(1).lower()[:4], "") != first for match in SECOND_ACTION.finditer(step))


# Preservation constraints are not editing goals: no step may say what stays the same. "Ensure ..." alone is not matched,
# since "ensure the hamburger has a sesame seed bun" is content.
KEEP_STEP = re.compile(
    r"(?i)^(?:keep|maintain|preserve|retain|leave)\b|\b(?:remains?|stays?|kept) (?:unchanged|the same|consistent|intact)\b"
    r"|\bunchanged\b"
)
# The same constraints in a source Correction, removed before measuring how much of it the steps cover.
_KEEP_SENTENCE = re.compile(r"(?i)^(?:keep|maintain|preserve|retain|leave)\b")
_ENSURE_SENTENCE = re.compile(r"(?i)^(?:ensure|make sure)\b.*\b(?:remains?|unchanged|the same|consistent|intact|unaltered)\b")
_KEEP_CLAUSE = re.compile(
    r"(?i),?\s*(?:while\s+|and\s+|but\s+)?(?:maintaining|keeping|preserving|retaining|leaving)\b[^.;]*"
    r"|,?\s*(?:while\s+|and\s+)?ensuring (?:that )?[^.;]*\b(?:remains?|unchanged|the same|consistent|intact)\b[^.;]*"
)
STEP_LEAK = re.compile(
    r"(?i)\b(?:first|second|original|edited|initial|previous) (?:image|picture|version|panel)\b|\b(?:has|have) been\b"
    r"|\bthe edit(?:ing)? operation\b|\bis replaced (?:by|with)\b|\bdiptych\b"
)

# ---------------------------------------------------------------------------------------------------------- prompt

SYSTEM_PROMPT = """You rewrite the critique and the edit instruction of an image-editing training example as numbered lists of editing goals.

Context. An image generation model drafts an image, critiques it against the caption ("Self-reflect"), writes an edit instruction ("Correction"), and then generates a corrected image. It will be tested on prompts that need edits like: set the number of an object type to an exact count, move one object relative to another, change one attribute of one object among several, render specific text, or remove something the prompt negates. It learns those fixes best when every instruction line is ONE editing goal, one action applied to one object, that it can compose with others. Your rewrite teaches that. A goal is not a sentence: do not split one goal into several lines, and do not leave two goals in one line.

The actions (always name the target object explicitly with its distinguishing attributes):
- add: "Add <object with all its attributes> <location relative to a named object or the frame>."
- remove: "Remove <object>."
- replace (swap): "Replace <object A> with <object B with all its attributes>."
- move: "Move <object> <relation> <reference object>." (left of, right of, above, below, behind, in front of, on, under, next to, into, out of, center, foreground, background). Taking an object out of one place and putting it in another ("remove the branch from the teapot and place it in a vase") is one move.
- change: modifies a target in place. Attributes ("Change the color of <object> from <old> to <new> and its material to <new>"), count ("Change the number of <objects> from <current> to <target> by adding/removing <k> <objects>", whenever the source states or implies both counts), pose/action/expression ("Change <subject> so that <subject> is <new pose, action, gaze or expression>"), text ("Change the text on <object> from "<old>" to "<new>""), and image-level properties, where the target is the image, the background, the sky or the scene (style, color palette, saturation, contrast, lighting, time of day, weather, season, viewpoint, crop).
Preservation constraints are not editing goals. Drop every sentence or clause that only says what must stay the same ("keep the background unchanged", "maintain the positions of all objects", "ensure the lighting remains consistent", "while maintaining the greenery of the forest"), and never write a step that keeps, maintains or preserves something.

Rules:
1. Faithfulness. Every concrete detail in the source (objects, counts, colors, positions, text strings, letter case, attribute values) must appear in exactly one step. Never add, infer, or drop content: no replacement object for a plain removal ("remove the hat" stays a removal), no location the Correction does not give, and no step for what must stay the same. Drop only preservation constraints and meta phrasing such as "The second image has been edited to", "The edit operation involves", "which makes the room feel ..." purpose clauses. Secondary adjustments such as "adjust the lighting and shadows to match" or "changing the palette to shades of brown and orange" are content, not meta phrasing: keep them, in the step of the target and action they belong to.
2. Sources stay separate. Correction steps come only from the Correction text; Self-reflect items come only from the Self-reflect text. Never turn a Self-reflect discrepancy that the Correction does not address into a correction step. The caption is only for resolving references; never copy words from it into a step.
3. Split and merge by goal. Start a new step only when the target object changes or the action changes. Everything the source does to the same target with the same action is ONE step, however many attributes, clauses or sentences it uses: "make the car red and shiny" is one change step; "reduce the saturation and increase the contrast of the image" is one change step; "add a red kite with a long tail" is one add step. Different targets or different actions are separate steps, and each step has exactly one action verb: "replace X, remove Y, add W" is three steps; "change the time of day to night and add a starry sky" is two steps (change, add); "add a calendar and rearrange the plates" is two steps; "remove the lamp and the rug" is two remove steps because they are two objects; two objects added at different places are two add steps. A set of objects the source edits as one unit, without giving each its own location or attributes ("replace the fruits with oranges, grapes and plums"), stays one step. Never invent locations, relations or sub-steps the source does not state.
4. Write every correction step as one imperative sentence (under 45 words), even when the source is descriptive or past tense ("has been removed" -> "Remove").
5. Resolve pronouns and vague references ("it", "them", "the second one") to concrete nouns, using the caption when needed.
6. Self-reflect items state one discrepancy each, grouped like the correction steps (one item per target and kind of problem, however many attributes are wrong), in the form "<what the image shows> instead of <what the caption asks>", "<object with its attributes> is missing", or "<object> should not be present". Keep them in the same order as the correction steps they motivate. A Self-reflect item describes only the current image against the caption: never mention a first, second, original or edited image, and never narrate an edit ("has been replaced", "is replaced by", "should have been removed"); compare against the caption instead ("fewer clouds" -> "the sky has few clouds instead of being cloudy").
7. Keep the operations in the order the source gives them.
8. If the source has a single editing goal, output a single step, even when that goal touches several attributes.
9. When the input says "Rewrite caption: yes", the caption was written as an edit of an earlier image. Also return "caption": the same caption rewritten as a standalone text-to-image prompt describing only the final image. Remove every reference to a previous state ("now", "no longer", "replacing the previous X", "instead of X", "has been removed", "compared to", "slightly reduced saturation", "increased contrast", "more square than rectangular") and state the target directly ("now features a gas cooktop" -> "features a gas cooktop"; "with reduced saturation" -> "with muted, desaturated colors"; "increased contrast" -> "high contrast"; "some tulips have been transformed into creatures" -> "some tulips are creatures"; "X is absent" or "X has been removed" -> "with no X"). Keep every other detail and the original wording where possible; do not add anything. Without that line, do not return a caption.

Respond with one JSON object and nothing else:
{"self_reflect": ["...", "..."], "correction": ["...", "..."]}
or, with "Rewrite caption: yes": {"caption": "...", "self_reflect": ["...", "..."], "correction": ["...", "..."]}
When the input has no Self-reflect, return "self_reflect": []."""

FEW_SHOT = [
    (
        {
            "caption": "A farmer is standing on the left side of the frame, hoeing the soil between rows of lush green crops. The farmer is wearing a red plaid shirt, green overalls, and brown boots. In the background, there is a vast landscape of rolling hills with a different pattern of crops featuring more pronounced green rows, and a sky with more clouds.",
            "self_reflect": "The farmer is positioned on the right side instead of the left, is kneeling and reaching into a basket instead of standing and hoeing, and the background shows rolling hills with different crop patterns and fewer clouds.",
            "correction": "The second image has been edited to show the farmer now on the left side of the frame, standing and hoeing the soil between rows of lush green crops. The farmer's attire remains the same, but the background landscape has been altered to show a different pattern of crops with more pronounced green rows, and the sky now includes more clouds, suggesting a different time of day or weather conditions.",
            "rewrite_caption": True,
        },
        {
            "caption": "A farmer is standing on the left side of the frame, hoeing the soil between rows of lush green crops. The farmer is wearing a red plaid shirt, green overalls, and brown boots. In the background, there is a vast landscape of rolling hills patterned with pronounced green crop rows, under a cloudy sky.",
            "self_reflect": [
                "The farmer is on the right side of the frame instead of the left side.",
                "The farmer is kneeling and reaching into a basket instead of standing and hoeing the soil.",
                "The crops on the rolling hills lack pronounced green rows.",
                "The sky has few clouds instead of being cloudy.",
            ],
            "correction": [
                "Move the farmer to the left side of the frame.",
                "Change the farmer so that the farmer is standing and hoeing the soil between rows of lush green crops.",
                "Change the pattern of the crops on the rolling hills of the background landscape to pronounced green rows.",
                "Add more clouds to the sky.",
            ],
        },
    ),
    (
        {
            "caption": "A group of three red-eyed tree frogs is perched on a lush green leaf with water droplets, in a misty forest with sun rays shining through.",
            "self_reflect": "The image shows five red-eyed tree frogs instead of three, with two frogs sitting on top of the middle frog.",
            "correction": "Remove the two frogs sitting on top of the middle frog.",
        },
        {
            "self_reflect": ["The image shows five red-eyed tree frogs instead of three; two of them sit on top of the middle frog."],
            "correction": ["Change the number of red-eyed tree frogs from five to three by removing the two frogs sitting on top of the middle frog."],
        },
    ),
    (
        {
            "caption": "An ornate Victorian armchair with intricate wooden carvings and red upholstery stands in an empty corner, with a large, ornate, arched mirror on the wall behind it.",
            "self_reflect": "The image features a Scandinavian-style wooden chair with a beige cushion instead of an ornate Victorian armchair with red upholstery. A potted plant and a small round table with a vase and a book are present, whereas they should be absent. The large, ornate, arched mirror on the wall behind the chair is missing.",
            "correction": "replace the Scandinavian-style chair with an ornate Victorian armchair with intricate wooden carvings and red upholstery, remove the potted plant and the small round table, and add a large, ornate, arched mirror on the wall behind the chair.",
        },
        {
            "self_reflect": [
                "The image shows a Scandinavian-style wooden chair with a beige cushion instead of an ornate Victorian armchair with red upholstery.",
                "A potted plant should not be present.",
                "A small round table with a vase and a book should not be present.",
                "The large, ornate, arched mirror on the wall behind the chair is missing.",
            ],
            "correction": [
                "Replace the Scandinavian-style wooden chair with an ornate Victorian armchair with intricate wooden carvings and red upholstery.",
                "Remove the potted plant.",
                "Remove the small round table with the vase and the book.",
                "Add a large, ornate, arched mirror on the wall behind the Victorian armchair.",
            ],
        },
    ),
    (
        {
            "caption": "A traditional Japanese tea set featuring a cast iron teapot, a matching trivet, a sugar pot, a creamer, and two sets of chopsticks wrapped in patterned fabric is arranged on a black lacquer tray on a tatami floor. A cherry blossom branch is placed in a tall vase behind the teapot.",
            "self_reflect": "The image displays a traditional tea set with a teapot, five cups, a bamboo whisk, and a bamboo scoop instead of a cast iron teapot, a matching trivet, a sugar pot, a creamer, and two sets of chopsticks wrapped in patterned fabric. Additionally, the cherry blossom branch is placed in the teapot rather than in a tall vase behind it.",
            "correction": "replace the tea set with a cast iron teapot, a matching trivet, a sugar pot, a creamer, and two sets of chopsticks wrapped in patterned fabric. Remove the cherry blossom branch from the teapot and place it in a tall vase behind the teapot.",
        },
        {
            "self_reflect": [
                "The image shows a teapot, five cups, a bamboo whisk and a bamboo scoop instead of a cast iron teapot, a matching trivet, a sugar pot, a creamer and two sets of chopsticks wrapped in patterned fabric.",
                "The cherry blossom branch is in the teapot instead of in a tall vase behind the teapot.",
            ],
            "correction": [
                "Replace the tea set (the teapot, five cups, bamboo whisk and bamboo scoop) with a cast iron teapot, a matching trivet, a sugar pot, a creamer and two sets of chopsticks wrapped in patterned fabric.",
                "Add a tall vase behind the cast iron teapot.",
                "Move the cherry blossom branch out of the teapot into the tall vase.",
            ],
        },
    ),
    (
        {
            "caption": "A farmers market stall with a sign reading 'Honey Honey', displaying oranges, blackberries and grapes, with jars of honey in the foreground and a warm orange and yellow color palette.",
            "self_reflect": "The sign reads 'Homemade Jams' instead of 'Honey Honey'. The display features strawberries, watermelons, and leafy greens instead of oranges, blackberries, and grapes. Jars of honey are missing from the foreground. The color palette lacks the intended warmer orange and yellow hues.",
            "correction": "The 'Homemade Jams' sign has been changed to 'Honey Honey'. The fruits and vegetables have been replaced with different types of fruits, such as oranges, blackberries, and grapes, and jars of honey are now visible in the foreground. The overall color palette seems warmer with more orange and yellow hues.",
        },
        {
            "self_reflect": [
                "The sign reads 'Homemade Jams' instead of 'Honey Honey'.",
                "The display shows strawberries, watermelons and leafy greens instead of oranges, blackberries and grapes.",
                "Jars of honey are missing from the foreground.",
                "The color palette lacks warm orange and yellow hues.",
            ],
            "correction": [
                "Change the text on the sign from 'Homemade Jams' to 'Honey Honey'.",
                "Replace the strawberries, watermelons and leafy greens with oranges, blackberries and grapes.",
                "Add jars of honey in the foreground.",
                "Change the color palette of the image to warmer orange and yellow hues.",
            ],
        },
    ),
    (
        {
            "caption": None,
            "self_reflect": None,
            "correction": "Add two fairies with light blue wings, one positioned slightly above and to the left of the male character's head, and the other slightly above and to the right of the female character's head. Ensure the fairies have green hair, purple tops, and green skirts, with small purple shoes. Maintain the overall lighting and composition of the scene.",
        },
        {
            "self_reflect": [],
            "correction": [
                "Add a fairy with light blue wings, green hair, a purple top, a green skirt and small purple shoes slightly above and to the left of the male character's head.",
                "Add a second fairy with light blue wings, green hair, a purple top, a green skirt and small purple shoes slightly above and to the right of the female character's head.",
            ],
        },
    ),
    (
        # Style, palette and shadows all change the image in place: one change step that keeps every clause (the model
        # used to drop the palette clause when it wrote the style alone).
        {
            "caption": None,
            "self_reflect": None,
            "correction": "Apply a watercolor filter to the entire image, changing the color palette from saturated reds and yellows to soft pastel blues and pinks. Maintain the positions and shapes of all objects, but soften the shadows to match the new palette.",
        },
        {
            "self_reflect": [],
            "correction": [
                "Change the image to a watercolor painting, with the color palette changed from saturated reds and yellows to soft pastel blues and pinks and the shadows softened to match.",
            ],
        },
    ),
    (
        {
            "caption": None,
            "self_reflect": None,
            "correction": "Change the color of the suit jacket from black to royal blue. Ensure the texture and lighting remain consistent with the original image.",
        },
        {
            "self_reflect": [],
            "correction": [
                "Change the color of the suit jacket from black to royal blue.",
            ],
        },
    ),
]


def _user_message(caption: str | None, self_reflect: str | None, correction: str, rewrite_caption: bool = False) -> str:
    parts = []
    if caption:
        parts.append(f"Caption: {caption}")
        if rewrite_caption:
            parts.append("Rewrite caption: yes")
    if self_reflect is not None:
        parts.append(f"Self-reflect: {self_reflect}")
    parts.append(f"Correction: {correction}")
    return "\n".join(parts)


def build_messages(job: dict) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_in, example_out in FEW_SHOT:
        messages.append({"role": "user", "content": _user_message(**example_in)})
        messages.append({"role": "assistant", "content": json.dumps(example_out, ensure_ascii=False)})
    messages.append({"role": "user", "content": _user_message(job["caption"], job["self_reflect"], job["correction"], job["rewrite_caption"])})
    if job.get("rejected"):
        # Retry: show the model its rejected reply and what was wrong with it.
        reply, reason = job["rejected"]
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"That answer was rejected: {reason} Answer again with the full JSON object."})
    return messages


def prompt_version(args) -> str:
    """Hash of everything that changes the reply, so a prompt edit invalidates the cache."""
    payload = json.dumps(
        [SYSTEM_PROMPT, FEW_SHOT, os.path.basename(args.model_path.rstrip("/")),
         args.temperature, args.top_p, args.top_k, args.presence_penalty],
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------- row parsing

def split_turn(text: str) -> dict | None:
    """Split a turn that carries a Correction (and optionally a Self-reflect) into its parts."""
    match = TURN_RE.match(text)
    if match is None or not match.group("corr").strip():
        return None
    return {
        "pre": match.group("pre"),
        "sr_marker": match.group("sr_marker"),
        "self_reflect": match.group("sr").strip() if match.group("sr_marker") else None,
        "corr_marker": match.group("corr_marker"),
        "correction": match.group("corr").strip(),
        "post": match.group("post") or "",
    }


def _as_list(conversations):
    if isinstance(conversations, str):
        return json.loads(conversations)
    if hasattr(conversations, "tolist"):
        return conversations.tolist()
    return list(conversations)


def find_edit_turn(conversations) -> tuple[int, dict] | None:
    """Index and parts of the turn that holds the Correction; the gpt turn wins if both do."""
    turns = _as_list(conversations)
    for role in ("gpt", "human"):
        for index, turn in enumerate(turns):
            if turn.get("from") == role and "orrection" in (turn.get("value") or ""):
                parts = split_turn(turn["value"])
                if parts is not None:
                    return index, parts
    return None


def caption_of(conversations) -> str | None:
    for turn in _as_list(conversations):
        if turn.get("from") == "human":
            value = (turn.get("value") or "").replace(IMAGE_TOKEN, " ")
            if "orrection" in value:  # UnicEdit: the human turn is the instruction, not a caption
                return None
            value = " ".join(value.split())
            return value or None
    return None


def caption_turn_index(conversations) -> int | None:
    """Index of the human turn that holds a plain-text caption (no image token, no instruction), which can be rewritten."""
    for index, turn in enumerate(_as_list(conversations)):
        if turn.get("from") == "human":
            value = turn.get("value") or ""
            if IMAGE_TOKEN in value or "orrection" in value or not value.strip():
                return None
            return index
    return None


def needs_caption_rewrite(conversations, caption: str | None) -> bool:
    return caption is not None and caption_turn_index(conversations) is not None and bool(CAPTION_DELTA.search(caption))


def junk_reason(parts: dict) -> str | None:
    text = f"{parts['self_reflect'] or ''}\n{parts['correction']}"
    for reason, pattern in JUNK_PATTERNS.items():
        if pattern.search(text):
            return reason
    return None


def job_key(version: str, job: dict) -> str:
    raw = json.dumps([version, job["caption"], job["self_reflect"], job["correction"]], ensure_ascii=False)
    return hashlib.sha1(raw.encode()).hexdigest()


# ------------------------------------------------------------------------------------------------- reply parsing

_STOP = set("a an the and or of to in on with for from by at is are be it its that this as into than then there their but".split())
# Instruction verbs and meta words the rewrite is told to normalise away ("has been edited to" -> "Change"); counting them
# made short sources fail coverage for faithful rewrites. Compared as 5-character stems.
_META = {w[:5] for w in (
    "edit edited editing image picture scene show shows showing make making ensure change changed changing adjust adjusted "
    "adjusting modify modified apply applied create creating transform transformed transforming second first overall entire "
    "seems appear appears appearance now more slightly while maintain maintaining keep keeping enhance enhancing additionally "
    "operation involves featuring feature resulting instead rather should would will been have has now also "
    # Narrated-edit wording ("the attire remains the same", "has been altered to show a different ...", "suggesting").
    "remains remain same altered alter different suggesting includes including"
).split()}
_LEADING_NUMBER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


def _content_stems(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w[:5] for w in words if len(w) > 2 and w not in _STOP and w[:5] not in _META}


def edit_text(correction: str) -> str:
    """The Correction without its preservation constraints, which the steps are told to drop."""
    sentences = re.split(r"(?<=[.!?])\s+", correction.strip())
    kept = [_KEEP_CLAUSE.sub("", sentence) for sentence in sentences
            if not _KEEP_SENTENCE.match(sentence) and not _ENSURE_SENTENCE.match(sentence)]
    text = " ".join(sentence for sentence in kept if sentence.strip())
    return text or correction


def coverage(original: str, steps: list[str]) -> float:
    """Share of the source's content-word stems that survive in the steps (a guard against dropped details)."""
    source = _content_stems(original)
    if not source:
        return 1.0
    return len(source & _content_stems(" ".join(steps))) / len(source)


def _clean_steps(value, name: str) -> tuple[list[str] | None, str | None]:
    """(steps, None) or (None, why the list was rejected)."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None, f'"{name}" must be a list of strings.'
    steps = []
    for item in value:
        item = _LEADING_NUMBER.sub("", " ".join(item.split()))
        if not item:
            continue
        if IMAGE_TOKEN in item or re.search(r"(?i)\b(self-reflect|correction)\s*:", item):
            return None, f'A {name} item contains a section marker or an image token: "{item}".'
        leak = STEP_LEAK.search(item)
        if leak:
            return None, (f'The {name} item "{item}" narrates the edit ("{leak.group(0)}"). Describe only the current '
                          f'image against the caption, or give an imperative step.')
        steps.append(item)
    return steps, None


def _clean_caption(value, original: str, args) -> tuple[str | None, str | None]:
    """A rewritten caption that no longer reads as an edit delta and neither drops nor invents much content."""
    if not isinstance(value, str) or not value.strip():
        return None, '"caption" is missing; return the rewritten caption.'
    value = " ".join(value.split())
    if IMAGE_TOKEN in value:
        return None, "The caption contains an image token."
    delta = CAPTION_DELTA_STRICT.search(value)
    if delta:
        return None, f'The caption still describes a change ("{delta.group(0)}"). State the final image directly.'
    if coverage(original, [value]) < args.min_caption_coverage:
        return None, "The caption rewrite drops details of the original caption; keep every detail of the final image."
    # Words in the rewrite that the original never had (guards against the model describing the draft or inventing).
    if coverage(value, [original]) < args.min_caption_precision:
        return None, "The caption rewrite adds content that is not in the original caption; do not add anything."
    return value, None


def parse_reply(text: str, job: dict, args) -> tuple[dict | None, str | None]:
    """(validated {"caption": str | None, "self_reflect": [...], "correction": [...]}, reason) or (None, reason).

    The reason says what failed validation; it is sent back to the model on the retry. "caption" is None when no
    rewrite was asked for or the rewrite failed validation; the steps are still usable, and the reason is set.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None, "The reply has no JSON object."
    try:
        reply = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None, "The reply is not valid JSON."
    if not isinstance(reply, dict):
        return None, "The reply is not a JSON object."
    correction, reason = _clean_steps(reply.get("correction"), "correction")
    if reason:
        return None, reason
    self_reflect, reason = _clean_steps(reply.get("self_reflect", []), "self_reflect")
    if reason:
        return None, reason
    if not correction:
        return None, '"correction" is empty.'
    if len(correction) > args.max_steps:
        return None, f"There are more than {args.max_steps} correction steps; merge steps that act on the same target with the same action."
    if job["self_reflect"] is not None:
        if not self_reflect or len(self_reflect) > args.max_steps:
            return None, f'"self_reflect" must have 1 to {args.max_steps} items.'
        # Details the critique drops are fine as long as a correction step still carries them.
        if coverage(job["self_reflect"], self_reflect + correction) < args.min_coverage:
            return None, "The self_reflect items drop details of the Self-reflect; keep every concrete detail."
    else:
        self_reflect = []
    if coverage(edit_text(job["correction"]), correction) < args.min_coverage:
        return None, "The correction steps drop details of the Correction; keep every concrete detail."
    for number, step in enumerate(correction, 1):
        if has_second_action(step):
            return None, (f'Correction step {number} ("{step}") joins two different actions. Split it into one step '
                          f"per action.")
        if KEEP_STEP.search(step):
            return None, (f'Correction step {number} ("{step}") says what should stay the same. Preservation '
                          f"constraints are not editing goals: remove that step or clause.")
    caption, reason = _clean_caption(reply.get("caption"), job["caption"], args) if job["rewrite_caption"] else (None, None)
    return {"caption": caption, "self_reflect": self_reflect, "correction": correction}, reason


def numbered(steps: list[str]) -> str:
    return "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))


def rebuild_turn(parts: dict, result: dict) -> str:
    text = parts["pre"]
    if parts["sr_marker"] is not None:
        text += f"{parts['sr_marker']}\n{numbered(result['self_reflect'])}\n"
    text += f"{parts['corr_marker']}\n{numbered(result['correction'])}"
    post = parts["post"]
    if post:
        text += "\n" + post.strip()
    return text


# --------------------------------------------------------------------------------------------------------- jobs

def source_column(parquet_file: pq.ParquetFile) -> str:
    return V1_COLUMN if V1_COLUMN in parquet_file.schema_arrow.names else SOURCE_COLUMN


def collect_jobs(files: list[Path], limit: int | None) -> tuple[list[dict], int, dict]:
    """One job per non-junk row that has a Correction, in a deterministic global order."""
    jobs, unparsed, junk = [], 0, {reason: 0 for reason in JUNK_PATTERNS}
    for file_index, path in enumerate(files):
        parquet_file = pq.ParquetFile(path)
        rows = parquet_file.read(columns=[source_column(parquet_file)]).column(0).to_pylist()
        for row_index, conversations in enumerate(rows):
            found = find_edit_turn(conversations)
            if found is None:
                unparsed += 1
                continue
            _, parts = found
            reason = junk_reason(parts)
            if reason is not None:
                junk[reason] += 1
                continue
            caption = caption_of(conversations)
            jobs.append({
                "file": file_index, "row": row_index, "caption": caption,
                "self_reflect": parts["self_reflect"], "correction": parts["correction"],
                "rewrite_caption": needs_caption_rewrite(conversations, caption),
            })
            if limit is not None and len(jobs) >= limit:
                break
        if limit is not None and len(jobs) >= limit:
            break
    return jobs, unparsed, junk


def load_cache(cache_dir: Path, version: str) -> dict:
    cache = {}
    for path in sorted(cache_dir.glob(f"{version}_rank*.jsonl")):
        with open(path) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:  # a line cut short by a killed run
                    continue
                cache[record["key"]] = record
    return cache


def run_llm(jobs, cache, cache_path: Path, model, tokenizer, args) -> None:
    """Generate every uncached job, retrying replies that fail validation, and append results to the cache."""
    no_think_kwargs = _resolve_no_think_kwargs(tokenizer)
    pending = [job for job in jobs if job["key"] not in cache]
    logger.info("rank %d: %d jobs, %d cached, %d to generate", args.rank, len(jobs), len(jobs) - len(pending), len(pending))
    # Sorting by source length keeps each batch's padding small.
    pending.sort(key=lambda job: len(job["correction"]) + len(job["self_reflect"] or ""))
    attempts = {job["key"]: 0 for job in pending}
    started, done = time.time(), 0
    batch_size = args.batch_size
    with open(cache_path, "a") as handle:
        while pending:
            batch, pending = pending[:batch_size], pending[batch_size:]
            try:
                replies = generate(model, tokenizer, [build_messages(job) for job in batch], args, no_think_kwargs, args.max_new_tokens)
            except torch.OutOfMemoryError:
                # Without the fused linear-attention kernels, prefill memory grows fast with batch x prompt length,
                # and batches get longer as the length-sorted queue advances. Halve and retry instead of dying.
                if batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                logger.warning("rank %d: CUDA OOM, batch size -> %d", args.rank, batch_size)
                pending = batch + pending
                continue
            for job, reply in zip(batch, replies):
                result, reason = parse_reply(reply, job, args)
                attempts[job["key"]] += 1
                caption_failed = result is not None and job["rewrite_caption"] and result["caption"] is None
                if (result is None or caption_failed) and attempts[job["key"]] <= args.max_retries:
                    job["rejected"] = (reply, reason)
                    pending.append(job)
                    continue
                record = {
                    "key": job["key"], "status": "ok" if result else "failed",
                    "caption": job["caption"], "self_reflect": job["self_reflect"], "correction": job["correction"],
                    "rewrite_caption": job["rewrite_caption"],
                    "result": result, "attempts": attempts[job["key"]], "raw": reply if result is None or caption_failed else None,
                    "reason": reason,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                cache[job["key"]] = record
                done += 1
            handle.flush()
            rate = done / max(time.time() - started, 1e-6)
            logger.info("rank %d: %d done, %d pending, %.2f rows/s", args.rank, done, len(pending), rate)


# ------------------------------------------------------------------------------------------------- shard rewrite

def new_conversations(conversations, cache, version, keep_junk: bool) -> tuple[list[dict], list[str]]:
    """New turns and the stats keys they count towards; a "dropped_*" key means the row is left out of the output."""
    turns = [dict(turn) for turn in _as_list(conversations)]
    found = find_edit_turn(turns)
    if found is None:
        return turns, ["unparsed"]
    index, parts = found
    reason = junk_reason(parts)
    if reason is not None:
        return turns, [f"kept_{reason}" if keep_junk else f"dropped_{reason}"]
    key = job_key(version, {"caption": caption_of(turns), "self_reflect": parts["self_reflect"], "correction": parts["correction"]})
    record = cache.get(key)
    if record is None or record["status"] != "ok":
        return turns, ["failed" if record else "missing"]
    turns[index]["value"] = rebuild_turn(parts, record["result"])
    statuses = ["ok"]
    if record.get("rewrite_caption"):
        caption = record["result"].get("caption")
        if caption is None:
            statuses.append("caption_kept")
        elif caption == record["caption"]:
            statuses.append("caption_unchanged_by_llm")
        else:
            turns[caption_turn_index(turns)]["value"] = caption
            statuses.append("caption_rewritten")
    return turns, statuses


def rewrite_file(path: Path, output_path: Path, cache: dict, version: str, keep_junk: bool) -> dict:
    parquet_file = pq.ParquetFile(path)
    src = source_column(parquet_file)
    conv_type = parquet_file.schema_arrow.field(src).type
    # Output order: everything as before, with the original conversations kept as v1 right next to the new ones.
    out_fields = []
    for field in parquet_file.schema_arrow:
        if field.name in (SOURCE_COLUMN, V1_COLUMN):
            continue
        out_fields.append(field)
    out_fields = [pa.field(SOURCE_COLUMN, conv_type), pa.field(V1_COLUMN, conv_type)] + out_fields
    # The pandas metadata block describes the old column set; drop it so to_pandas() does not rely on it.
    metadata = {k: v for k, v in (parquet_file.schema_arrow.metadata or {}).items() if k != b"pandas"}
    schema = pa.schema(out_fields, metadata=metadata or None)
    compression = parquet_file.metadata.row_group(0).column(0).compression.lower() if parquet_file.metadata.num_row_groups else "snappy"

    stats = {"ok": 0, "failed": 0, "missing": 0, "unparsed": 0}
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    writer = pq.ParquetWriter(temporary, schema, compression=compression)
    try:
        for group in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(group)
            source = table.column(src).to_pylist()
            rebuilt, keep = [], []
            for conversations in source:
                turns, statuses = new_conversations(conversations, cache, version, keep_junk)
                for status in statuses:
                    stats[status] = stats.get(status, 0) + 1
                if statuses[0].startswith("dropped_"):
                    keep.append(False)
                    continue
                keep.append(True)
                rebuilt.append(turns)
            if not all(keep):
                table = table.filter(pa.array(keep))
            if table.num_rows == 0:
                continue
            columns = {name: table.column(name) for name in table.column_names if name not in (SOURCE_COLUMN, V1_COLUMN)}
            columns[SOURCE_COLUMN] = pa.array(rebuilt, type=conv_type)
            columns[V1_COLUMN] = table.column(src)
            writer.write_table(pa.table({f.name: columns[f.name] for f in schema}, schema=schema), row_group_size=table.num_rows)
        writer.close()
        writer = None
        os.replace(temporary, output_path)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    return stats


# ----------------------------------------------------------------------------------------------------------- main

def wait_for_ranks(cache_dir: Path, version: str, args) -> None:
    (cache_dir / f"{version}_rank{args.rank}.done").touch()
    while True:
        done = sum((cache_dir / f"{version}_rank{r}.done").exists() for r in range(args.world_size))
        if done == args.world_size:
            return
        logger.info("rank %d: waiting for %d/%d ranks to finish generating", args.rank, args.world_size - done, args.world_size)
        time.sleep(60)


def process_dir(input_dir: Path, model, tokenizer, args, version: str) -> None:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {input_dir}")
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else input_dir / ".decompose_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    jobs, unparsed, junk = collect_jobs(files, args.limit)
    for job in jobs:
        job["key"] = job_key(version, job)
    # Rows with the same text share one key and one cache record; generate each key once (before the rank split, so
    # two ranks never generate the same key).
    seen = set()
    unique = [job for job in jobs if not (job["key"] in seen or seen.add(job["key"]))]
    logger.info("%s: %d files, %d rows with a correction (%d unique, %d flagged for a caption rewrite), %d rows without "
                "(left unchanged), junk rows %s (%s)", input_dir, len(files), len(jobs), len(unique),
                sum(job["rewrite_caption"] for job in unique), unparsed, junk, "kept unchanged" if args.keep_junk else "dropped")
    my_jobs = [job for index, job in enumerate(unique) if index % args.world_size == args.rank]

    if args.stage != "rewrite":
        # A marker left by an earlier run must not tell the other ranks this one is done before it has generated.
        (cache_dir / f"{version}_rank{args.rank}.done").unlink(missing_ok=True)
        cache = load_cache(cache_dir, version)
        run_llm(my_jobs, cache, cache_dir / f"{version}_rank{args.rank}.jsonl", model, tokenizer, args)

    if args.dry_run:
        for job in my_jobs:
            record = cache[job["key"]]
            print("=" * 100)
            if job["caption"]:
                print("CAPTION     :", job["caption"])
            if job["self_reflect"] is not None:
                print("SELF-REFLECT:", job["self_reflect"])
            print("CORRECTION  :", job["correction"])
            print("-" * 40, record["status"], f"(attempts={record['attempts']})")
            if record["result"]:
                if job["rewrite_caption"]:
                    print("Caption:", record["result"]["caption"] or f"<rewrite failed, original kept> RAW: {record['raw']}")
                if record["result"]["self_reflect"]:
                    print("Self-reflect:\n" + numbered(record["result"]["self_reflect"]))
                print("Correction:\n" + numbered(record["result"]["correction"]))
            else:
                print("RAW REPLY:", record["raw"])
        return

    if args.stage == "generate":
        logger.info("rank %d, %s: generation done (--stage generate, shards not rewritten)", args.rank, input_dir.name)
        return
    if args.stage == "rewrite":
        cache = load_cache(cache_dir, version)
        missing = sum(job["key"] not in cache for job in unique)
        if missing:
            logger.warning("%s: %d of %d unique rows have no cache record and keep their original text", input_dir.name,
                           missing, len(unique))
    elif args.world_size > 1:
        wait_for_ranks(cache_dir, version, args)
        cache = load_cache(cache_dir, version)
    output_dir = Path(args.output_dir).expanduser() / input_dir.name if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    my_files = [path for file_index, path in enumerate(files) if file_index % args.world_size == args.rank]

    def rewrite(path: Path) -> dict:
        stats = rewrite_file(path, output_dir / path.name, cache, version, args.keep_junk)
        logger.info("%s -> %s: %s", path.name, output_dir / path.name, stats)
        return stats

    # Shard rewrites are I/O and (de)compression bound, which pyarrow runs outside the GIL: threads overlap them.
    totals = {}
    with ThreadPoolExecutor(max_workers=max(1, args.rewrite_workers)) as pool:
        for stats in tqdm(pool.map(rewrite, my_files), total=len(my_files), desc=f"rank {args.rank} rewrite",
                          unit="shard", position=int(os.environ.get("LOCAL_RANK", 0)), dynamic_ncols=True):
            for name, value in stats.items():
                totals[name] = totals.get(name, 0) + value
    logger.info("rank %d, %s: rows rewritten %s", args.rank, input_dir.name, totals)


def main(args) -> None:
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s rank{args.rank} %(levelname)s %(message)s")
    version = prompt_version(args)
    logger.info("prompt version %s", version)
    model, tokenizer = load_model(args) if args.stage != "rewrite" else (None, None)
    for input_dir in args.input_dirs:
        process_dir(Path(input_dir).expanduser(), model, tokenizer, args, version)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Decompose Self-reflect / Correction text into numbered atomic steps.")
    parser.add_argument("--input_dirs", nargs="+", default=input_dirs)
    parser.add_argument("--output_dir", default=None,
                        help="Write shards to <output_dir>/<input dir name>/ instead of in place (default: in place).")
    parser.add_argument("--cache_dir", default=None, help="Default: <input dir>/.decompose_cache")
    parser.add_argument("--model_path", default=llm_model_path)
    parser.add_argument("--device", default="auto", help="`device_map` for from_pretrained.")
    parser.add_argument("--attn_implementation", default=None, help="e.g. sdpa or flash_attention_2")
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Start size; halved automatically on CUDA OOM (12 fits one A100-80G with the torch fallback kernels).")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    # Qwen3.6 model card, instruct (non-thinking) mode.
    parser.add_argument("--temperature", type=float, default=0.7, help="0 for greedy decoding.")
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=1.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=12, help="Reject replies with more steps than this.")
    parser.add_argument("--min_coverage", type=float, default=0.6,
                        help="Reject replies that keep less than this share of the source's content words.")
    parser.add_argument("--min_caption_coverage", type=float, default=0.5,
                        help="Reject caption rewrites that keep less than this share of the caption's content words "
                             "(lower than --min_coverage: the delta wording about the previous image is meant to go).")
    parser.add_argument("--min_caption_precision", type=float, default=0.8,
                        help="Reject caption rewrites whose content words are less than this share found in the original.")
    parser.add_argument("--keep_junk", action="store_true",
                        help="Keep junk rows (placeholders, no-edit replies, diptych critiques) unchanged instead of dropping them.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N rows per input dir.")
    parser.add_argument("--dry_run", action="store_true", help="Print before/after and do not write shards.")
    parser.add_argument("--stage", choices=("all", "generate", "rewrite"), default="all",
                        help="all: generate, wait for every rank, rewrite shards. generate: only fill the cache (no rank "
                             "barrier, so ranks can run as separate jobs). rewrite: only rewrite shards from the cache "
                             "(no model, CPU only).")
    parser.add_argument("--rewrite_workers", type=int, default=4, help="Shards rewritten concurrently per rank.")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    args = parser.parse_args(argv)
    if args.dry_run and args.stage == "rewrite":
        parser.error("--dry_run prints generations; it does nothing with --stage rewrite")
    if args.limit is not None and not args.dry_run and args.output_dir is None:
        parser.error("--limit without --dry_run would rewrite shards in place with most rows unprocessed; "
                     "add --dry_run or --output_dir")
    return args


if __name__ == "__main__":
    main(parse_args())
