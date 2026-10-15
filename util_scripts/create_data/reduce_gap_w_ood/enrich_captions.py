'''
Script to enrich the captions to include details that are required in the test sets - Geneval2, TIIF and Genai-bench. Specifically,
run a pipeline of specialized models to extract the information from an image like count of objects, 2D/3D relative positions,
specific object's attributes, etc. (refer to the examples below to get all the categories) and augment the caption(s) in the given
dataset.

For grounded caption generation pipeline, refer https://github.com/remyxai/VQASynth and https://github.com/AnjieCheng/SpatialRGPT


Test sets and their prompts / captions' breakdown:
## GenEval2 (per-atom judge)

**Count (609 flagged prompts with a failed count atom, 14% fixed)**
- "three spotted flamingos, and seven glass turtles under seven black bicycles". Failed: how many flamingos, turtles, bicycles. Model: "only five bicycles instead of seven. Correction: Add two more bicycles to make a total of seven". Result: 0 of 3 fixed, AM 40 to 40.
- "a white giraffe, and four glass cookies to the right of a croissant". Failed: how many cookies. Model: "two glass rings, not four glass cookies. Correction: Replace the two glass rings with four glass cookies". Result: not fixed.
- "two spotted giraffes in front of a green dog playing with three checkered horses". Failed: dogs, horses counts. Model asks for a checkered pattern and a green dog. Result: 0 of 2, AM 38 to 30.

Needed edit: set the number of an existing object type to N (add or remove k of the same object while keeping the rest), with N from 3 to 7 and often two counted types at once.

**Position (312 prompts, 9 to 12% fixed)**
- "a striped monkey to the right of a mushroom to the right of a koala". Failed: monkey right of mushroom, mushroom right of koala. Model: "Replace the koala with a striped monkey and move it to the ...". Result: 0 of 2, AM 49 to 23.
- "six glass trucks to the right of six elephants, and six metal lions". Model: "Reduce the number of elephants to six, add six glass trucks to the right, and include six metal lions". Result: not fixed.
- "three wooden croissants under seven wooden bears to the left of a metal lion". Model: "Add two more wooden bears... add three wooden croissants to the left of the bears". Result: 0 of 2.

Needed edit: place object A relative to object B (right of, under, left of), usually chained across three objects, combined with count changes.

**Attribute (414 prompts, 11 to 13% fixed)**
- "a yellow violin to the right of a purple rabbit chasing five sparkling turtles". Failed: rabbit purple, turtles sparkling. Model: "Change the violin to yellow, ...". Result: 0 of 2, AM 52 to 44.
- "a red koala, and six blue cats behind six red mushrooms". Failed: koala red, cats blue. Result: 0 of 2.
- "a black dog, and six yellow birds behind seven glass clocks". Failed: clocks glass. Model only asks to add one bird and one clock. Result: not fixed, AM 60 to 50.

Needed edit: recolor or change material of one counted group without touching the others (blue cats but red mushrooms). Recolor is the best-trained op, yet it fails when the attribute is bound to a multi-object group.

**Object (326 prompts, 25% fixed, the best skill)**
- "a purple umbrella to the left of four striped guitars to the left of three sparkling violins". Failed: no guitars. Model: "Replace the violins with four striped guitars". Result: fixed, but AM 56 to 46 because the violins are now gone.
- "six pigs and a croissant and six trucks". Failed: no croissant. Model only asks for one more truck. Result: not fixed, AM 50 to 33.

Needed edit: add a missing object type into an already crowded scene without displacing existing ones.

## TIIF (per-question yes/no judge)

**Numeracy (flagged edits net negative)**
- "[short] four helmets, a single lamp, one sheep, and four bowls". Failed: bowls = 4. Model: "five helmets instead of four, two bowls instead of four. Correction: Add one more helmet and two more bowls". Result: still wrong, accuracy 0.88 to 0.75.
- "[short] one television surrounded by three camels and four chickens". Model: "Change the number of camels to three and the number of chickens to four". Result: still wrong, 0.83 to 0.50.

Needed edit: adjust two counts simultaneously, one up and one down.

**2d and 3d spatial relation (net negative)**
- "[short] A couch is positioned to the right of a fish". Model: "move the fish to the right of the couch". Result: unchanged.
- "[long] ...to the left of the pig ... the unassuming cup...". Model: "Move the cup to the left of the pig". Result: unchanged.
- "[short] A balloon sits behind the desk". Model contradicts itself: "Move the balloon to be in front of the desk, not behind it". Result: unchanged.

Needed edit: swap the left/right or front/behind relation of two existing objects. The 3d example also shows the critic getting depth relations backwards.

**Negation (net negative, −12 on the negation dimension)**
- "[short] The fork is not piercing the steak, remaining above it". Model: "Adjust the fork so it is positioned above the steak without...". Result: unchanged.
- "[short] the windbreaker is not positioned over the sweater". Model asks for the opposite (layer it over). Result: unchanged.

Needed edit: change a physical relation to its negated form, or re-arrange two garments; the critic sometimes inverts the negation.

**Text (23% of TIIF failures, +8 when flagged)**
- Engine with text "ICT", "Billet", "LLC", "Performance", "Engineering", "Power", "Precision". Model: "add text on the engine, including...". Result: 3 of 7 words now rendered, 0.00 to 0.43.
- Diamond logo with nine words. Model: "Change the text to ...". Result: 0.22 to 0.11.

Needed edit: render or correct multiple specific words. Adding text helps somewhat; rewriting existing text does not.

**Dimensions where edits helped**
- shape+color: "conical tennis hat". Model: "Replace the Asian hat with a tennis hat". Result: 0.60 to 1.00. Pure replace-object, the most-trained op.
- comparison+3d: "smaller mouse on the table, larger mouse hangs over the side". Model asks for a size difference. Result: 0.50 to 0.75.
- style: "Disney style" fixed after edit (0.00 to 1.00); "LEGO style" not fixed.

## GenAI-bench (prompt-level VQAScore, keyword tags)

- **Counting / comparison.** "Three mugs side by side; the two closest to the faucet each contain a toothbrush, the furthest is empty". Model: "Move the toothbrush to the furthest mug and ensure the other two are empty". Result: 0.00 to 0.00. "Balls on the table more than on the floor (three each)". Model: "Increase the number of balls on the table". Result: 0.01 to 0.00. Needed edit: change a relative count between two groups.
- **Negation.** "A garden with flowers, but no bees". Model: "remove the bees". Result: 0.46 to 0.36. "A cozy bedroom without a fluffy pillow". Model: "remove the fluffy pillow". Result: 0.05 to 0.01. Needed edit: remove a named object. This op is trained (12% of the mix) and the critic gets it right, yet the score does not move, which points at execution on its own drafts rather than the vocabulary.
- **Spatial.** "A plant to the left of a window". Model: "move the plant to the left of the window". Result: 0.27 to 0.29.
- **Action and scene.** "A cow is being milked at dawn". Model asks for dawn lighting only. Result: 0.00 to 0.00, the milking action is never addressed. "A cowboy rides a mechanical bull in Times Square". Model: "Replace the bull with a mechanical bull". Result: 0.36 to 0.93, the one clear success in the sample and again a replace-object edit.
- **Attribute on a specific instance.** "Two cars, the one on the left is red, the one on the right is not". Model: "Change the color of the car on the right". Result: 0.00 to 0.00. Needed edit: recolor one instance of a pair, selected by position.

**Pattern across all three sets.** The edits that succeeded in these samples are all replace-one-object (tennis hat, mechanical bull, guitars for violins). The edits that fail are: set a count on a group, move A relative to B, recolor one instance out of a group, remove an object from the model's own draft, and render specific text. Those are the operations to synthesize refine pairs for.


Input data:
    Columns:
        conversations:
            human: {caption}
            gpt: <image-draft>\nSelf-reflect:...\nCorrection:...<image>

        conversations_short:    # optional
            human: {short_caption}
            gpt: <image-draft>\nSelf-reflect:...\nCorrection:...<image>


{caption} and {short_caption} describe <image>

For the output, add a new column "conversations" and "short_conversations", and rename the current "conversations" and
"short_conversations" to "v1-conversations" and "v1-short_conversations".

Ouput:
    Columns:
        conversations:
            human: {augmented_caption}
            gpt: <image-draft>\nSelf-reflect:{augmented_self_reflect}\nCorrection:{augmented_correction}<image>

        conversations_short:    # optional
            human: {augmented_short_caption}
            gpt: <image-draft>\nSelf-reflect:{augmented_self_reflect}\nCorrection:{augmented_correction}<image>

The Self-reflect and the Correction carry the facts the caption took about the objects they already name (a count,
attribute, direction or text written next to that noun: "replace the zebras with a lion facing left"); they never
gain an object or a clause (--no_critique keeps them as they were). Rows without a critique keep their gpt turn.


Don't use vllm yet, would like to iterate faster for now by avoiding vllm launch overhead.

--------------------------------------------------------------------------------------------------------------------
Implementation notes
--------------------------------------------------------------------------------------------------------------------

Per row (caption, draft image, Self-reflect, Correction, final image), all on one GPU, no vLLM. Only the final image
(the one the caption describes) is measured (see D). Checked on 47 rows by measuring both images: all 51 counts
that were reliable on both agreed.

    A. Vocabulary (Qwen3.6-27B, final image + caption): up to 10 countable object kinds as detector queries
       (singular, plural, kind person/animal/vehicle/object) and 2-3 plausible objects that are NOT there
       (negation candidates).
    B. Perception on the final image, four models (2026-10-09: Florence-2, OWLv2, SAM 2 and Orient-Anything were
       removed; in the 72-row v9 review they decided no count fact and fed only facing, part, garment, hair and
       text facts, while taking most of the perception time):
         - SAM 3 (concept segmentation, one noun per query): instances with masks and scores; the primary counter
           and the mask source. Grounding DINO (one query per object kind, fp16): boxes, the second vote. Per kind
           and per detector: NMS, drop "group" boxes that contain two or more other boxes of the same kind (the
           classic "one box around the whole herd" failure), drop boxes nested in a larger box of the same kind,
           one box one noun.
         - Count vote (count_vote): SAM 3 and Grounding DINO must agree, and SAM 3's instances are the objects.
           One borderline SAM 3 instance next to two or more clean ones is settled by that agreement; more
           borderline instances make the count unreliable. SAM 3 is no longer overruled (that needed Florence-2
           as a third vote), and a lone clean SAM 3 instance with a faint Grounding DINO box on it is one object.
         - Depth Anything V2: per-instance depth as the 80th percentile of inverse depth over its mask minus the
           masks it contains (the nearest surface of the shelf, not of the pots on it), for in front of / behind.
         - SigLIP2 on each masked instance (pasted on white and on black, both must agree): color, material, pattern.
         - Negation: a candidate is absent when SAM 3 returns no instance of it at all.
    C. Regions (Qwen3.6-27B, scene + crops of up to --max_regions singletons and left/right pairs other than
       people, i.e. objects a caption can name unambiguously): color and material as JSON. A claim becomes a fact
       only when it is among SigLIP's two best on the object's mask (SpatialRGPT / VQASynth caption regions; here
       the region captions are verified instead of trusted).
    D. Facts: exact counts, 2D relations (left/right/above/below between whole groups, none when one spans the
       other), 3D relations (in front of/behind), "on" (the upper mask's lower edge touches the lower mask),
       group-bound colors ("all four cats are blue"), instance-bound colors ("the car on the left is red"), the
       verified region colors and materials, comparisons and verified absences ("no bees").
       Positions are always relative to other objects, never to the frame ("to the left of the woman", not "on
       the left side of the image").
       Up to max_facts facts are kept per row in priority order: count, position (2D, 3D, on), attribute,
       comparison, negation (negations only in a --negation_prob share of rows, one at
       most).
       Only the final image is measured: the draft differs from it only in what the Self-reflect states, so every
       measured fact the critique does not talk about holds for the draft too, and the critique only gains facts
       about objects it already names (stage E). The rewrite skips any fact that contradicts the caption, the
       Self-reflect or the Correction.
    E. Rewrite (Qwen3.6-27B, final image + caption + short caption + critique + selected facts): JSON with the new
       caption(s), the Self-reflect and Correction enriched with the same facts, and the fact ids used. Validated
       (JSON, no <image>, the original's words and quoted text kept, bounded growth, used count facts carry their
       number) and retried, also when it used fewer than --min_used_fraction of the facts. The Self-reflect and
       Correction are validated the same way (words kept, bounded growth, only the facts' words, no object the
       original does not name, no "keeping ..." clauses); after the retries a rewrite whose critique still fails
       keeps the caption and the original critique.

    F. Veto (Qwen3.6-27B, final image + the selected facts; --no_veto disables): each fact is answered yes / no /
       unsure from the image; a "no" drops the fact. The VLM only removes facts, never adds or changes one, and
       count facts are exempt unless --veto_counts (VLMs miscount). Two facts more than max_facts are selected so
       that a veto still leaves a full row.
    G. Short captions (--synth_short): datasets without a short-caption column get "conversations_short" with a
       prompt-style caption written from the used facts ("seven lions in front of an acacia tree, with no
       zebras"), the style of the test prompts; the loader (llava/data/dataset.py) samples it half the time. Rows
       without one carry a copy of the enriched conversations, so the column never holds a null.

Precision rules learnt from the first runs (each one produced a wrong caption in the 47-row sample):
    - a box lying inside a larger box of the same noun is the same object ("two suns": core + halo; "two towers":
      the top half and the whole) -> clean_boxes drops it; sun/moon/sky are single by definition (UNIQUE_OBJECTS)
    - a box that two nouns claim ("book" and "newspaper" on the stack of newspapers) goes to the better-scoring
      noun and makes the other's count unreliable (_assign_boxes)
    - as many borderline boxes as strong ones is a forest, not "five trees"; boxes under --min_box_frac of the
      image are not counted (count_vote / analyze)
    - "on" needs the top object's bottom edge inside the bottom object's box, its centre over it, and the same
      depth on both sides of the contact (a wall clock above a sofa touched the sofa's mask); nothing is on a
      house or on the sun (_rests_on)
    - in front of / behind needs the two instances to line up horizontally and not to lie one inside the other
      (the journal "in front of" the pens at the table's edges, the table "in front of" its rug) (relation_3d)
    - at most --max_position_facts position facts and --max_3d_facts depth facts per row, pairs the caption names
      first, left/right before on before in front of/behind (select_facts): a still life is not "behind the
      saucer, in front of the doily, with a curtain behind the cat and behind the table"
    - a count the caption already states is not offered ("reading a book" became "reading one of two books"), and
      when it contradicts the detectors the noun takes part in no fact at all (the second "book" was the stack of
      newspapers); rewrites with "totaling", "in number", "one of two" are rejected, as are rewrites whose new
      content words come from neither the caption nor the facts ("A woman handles a black plastic record near a
      box" for two attribute facts about the record); garments are never placeable objects ("the tower is behind
      the vest"); an attribute equal to the noun is dropped ("the paper is paper"); buildings and rooms get no
      region description; a relation the caption states ("a lamp to the right of the sofa") must survive the
      rewrite; a rewrite with filler ("specifically two curtains") is rejected
    - a number the caption states wins over the detectors, except when both counters agree on another one with
      no borderline box ("a wooden-legged chair" over two chairs, "three
      women and one woman" over five, "a fishing rod" over two): then the fact tells the rewrite to replace the
      caption's number, and the VLM veto checks that one fact. An article counts only its own singular noun ("a
      park bench feeding pigeons" states no number of pigeons). A count the rewrite wrote as a digit is spelled out

Results go to a JSONL cache keyed by (prompt/threshold version, row id, source conversations), so a re-run skips
finished rows. The shard rewrite is the same as decompose_reflection_correction.py: row-group sizes and codec kept,
atomic replace, "<col>" -> "v1-<col>" with the enriched copy written as "<col>"; if "v1-<col>" already exists (an
earlier run) it is the source. Only the human turn changes; rows without a caption, without facts or whose reply
never validated keep their original text.

NOTE: decompose_reflection_correction.py uses the same "v1-conversations" name. Running it first and this script
second would make this script enrich the ORIGINAL (non-decomposed) conversations; pass --v1_prefix to keep both.

Multi-GPU: one process per GPU (enrich_captions.sh). Rank i processes global row groups i, i+N, ..., writes a
done-marker, waits for all ranks, then rewrites shards i, i+N, ...

Quick iteration: --dry_run --limit 16 [--viz_dir DIR] prints before/after and the facts, writes no shards
(facts marked * were selected, x were vetoed). util_scripts/sft_data_convert_format/viz/enrich_caption.py draws a
sheet per row from the cache.
'''

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw

# Model loading, the presence-penalty logits processor and the no-think probe are shared with the caption-error
# script; row parsing and the content-word coverage check with the decompose script next to this one.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "add_errors_in_caption"))
from add_errors_in_caption_hf import PresencePenalty, _resolve_no_think_kwargs, load_model  # noqa: E402
from decompose_reflection_correction import _as_list, coverage, find_edit_turn  # noqa: E402

logger = logging.getLogger("enrich")

# actual data to run on
# "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/gpt-edit-simpler_hqedt_slfref_tar/train",
# "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/gpt-edit-simpler_hqedt2_slfref_tar/train",

input_dirs = [
    "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/gpt-edit-simpler_hqedt_slfref_tar/small_sample",
    "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/gpt-edit-simpler_hqedt2_slfref_tar/small_sample",
]

# Refer the model card https://huggingface.co/Qwen/Qwen3.6-27B for input preparation and sampling parameters
# Model to generate captions grounded in the information extracted using specialized models as a mitigation for VLMs'
# high error rate for things like counting, relative position, etc.
llm_model_path = "/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/Qwen3.6-27B"

# download models for the pipeline in this dir
models_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/caption_aug_models"
# MM-Grounding-DINO (openmmlab-community/mm_grounding_dino_large_all) returns flattened boxes under Transformers
# 5.14, so the original Grounding DINO is used.
detector_path = f"{models_dir}/grounding-dino-base"                     # IDEA-Research/grounding-dino-base
depth_path = f"{models_dir}/Depth-Anything-V2-Large-hf"                 # depth-anything/Depth-Anything-V2-Large-hf
sam3_path = f"{models_dir}/sam3"                                        # facebook/sam3 (concept segmentation)
attribute_path = "/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/siglip2-so400m-patch14-384"

CONV_COLUMN = "conversations"
SHORT_COLUMNS = ("conversations_short", "short_conversations")  # the spec uses both spellings
IMAGE_COLUMN = "image"
ID_COLUMN = "id"
IMAGE_TOKEN = "<image>"

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
# Bumped when a rule changes what a row produces without any prompt or threshold changing, so the cache is redone.
RULES_VERSION = "2026-10-09 perception: SAM 3 + Grounding DINO + Depth Anything + SigLIP only"
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray", "silver", "gold", "beige"]
MATERIALS = ["real", "wooden", "metal", "glass", "plastic", "stone", "ceramic", "paper", "fabric", "leather"]
# "sparkling" was dropped: SigLIP gave it >0.7 on ordinary leather chairs.
PATTERNS = ["plain", "striped", "spotted", "checkered", "floral"]
NEUTRAL = {"real", "plain"}

# ---------------------------------------------------------------------------------------------------------- prompts

VOCAB_PROMPT = """You prepare queries for an open-vocabulary object detector that will measure this image.

Caption of the image: {caption}

Return one JSON object on a single line (no indentation, no code fence) and nothing else:
{{"objects": [{{"name": "<singular noun>", "plural": "<plural noun>", "kind": "person" | "animal" | "vehicle" | "object"}}, ...], "absent": ["<noun>", ...]}}

- objects: up to 10 distinct kinds of discrete, countable things that are clearly visible, including small and background ones (flower pots on a shelf, pictures on a wall, a wall clock, a lamp, a cutting board): people, animals, vehicles, furniture, tools, food items, individual plants (a tree, a flower), buildings, signs. Use the simplest common noun a detector understands ("lion", "woman", "cup", "tree", "bread slice"), lowercase, no colors, materials or counts. Include the countable things the caption names when they are visible. Never list one thing twice under two names (not both "person" and "woman"). Skip surfaces, regions and stuff (sky, ground, grass, snow, water, wall, floor, forest, mountain range, background), parts of another listed object and things someone wears (legs, wheels, a handle, glasses, a jacket), and things too many to count (leaves, snowflakes, stars).
- kind: "person" for people, "animal", "vehicle" (cars, trains, bicycles, boats, planes, tractors), otherwise "object".
- absent: up to 3 specific objects (not categories such as "animal" or "person") that would be plausible in this scene, so that a prompt might ask for the scene without them ("a garden with no bees"), but are NOT in the image and are not mentioned in the caption."""

WRITE_SYSTEM_PROMPT = """You enrich an image caption with verified facts about the image.

Context. The caption describes the image shown. A text-to-image model is trained on it and later tested on prompts that pin down exact counts ("three spotted flamingos and seven glass turtles"), relative positions ("a monkey to the right of a mushroom", "a balloon behind the desk"), attributes bound to one group or one instance ("six blue cats behind six red mushrooms", "the car on the left is red"), comparisons ("the mouse on the left is larger") and negations ("a garden with flowers but no bees"). The facts were measured with object detectors, segmentation and a depth estimator, and each was cross-checked between models, so trust them over your own impression of counts, positions and directions.

The Self-reflect and Correction describe how an earlier draft of this image differed from the caption. Never copy their wording into the caption; rule 8 says how to enrich them.

Rules:
1. Keep every word and detail of the original caption: insert, do not rewrite. The only words you may replace are vague quantities ("a group of", "several", "a pride of lions"), with the count from a fact. A count the caption states, including a singular "a"/"an", wins over a count fact: skip that fact, unless the fact says the caption's number is wrong: then replace the caption's number with the fact's. Never make a singular noun of the caption refer to one of several ("reading a book" stays "reading a book", never "reading one of two books").
2. Work every fact you use into the caption, attached to the object it is about, with the phrasing such prompts use: number words right where the group is first named ("seven lions rest", never "7 lions", never "exactly", never an afterthought such as "totaling seven lions" or "a total of seven"), "to the left of", "to the right of", "above", "below", "in front of", "behind", "on", attributes right before the noun ("two blue cats", "a wooden chair"), negation as "with no ..." or "without any ...".
3. Text the caption already quotes stays exactly as the caption writes it (letter case included). Skip a fact that contradicts the caption, the Self-reflect or the Correction, or that is plainly wrong in the image (the object named is not what is shown). Do not skip a fact only because you would count or place things differently.
4. Add nothing beyond the caption and the facts: no new objects, colors, moods or style words. Skip a fact the caption already states in other words ("a bronze teapot" already says metal, "an oak chair" already says wooden).
5. The result reads as one natural caption in the register of the original. It may grow by a few clauses or a sentence or two to hold the facts; do not turn it into a list of facts. Use as many facts as fit, but attach at most one position clause to a noun in a sentence, and leave the caption's own phrases where they are.
6. If a short caption is given, return an enriched short caption too: keep every word of the given short caption in its place and insert at most two of the facts you used (preferring counts and relations) as a word or a short clause, exactly as for the long caption. Never rewrite or shorten it.
7. If the short caption line says "(none: write one)", write a new short caption instead: one sentence of at most {short_max_words} words in the style of a text-to-image prompt, naming the main objects of the caption with the counts, positions and attributes from the facts you used ("seven lions in front of an acacia tree in a snowy landscape, with no zebras"). Nothing beyond the caption and the facts.
8. If a Self-reflect and a Correction are given, return both with the facts you used in the caption written onto the objects they ALREADY name, so that they match the enriched caption: a count or attribute goes right next to that noun where it stands ("replace the zebras with two lions", "add a wooden chalkboard sign", "the two blue cats are missing"), in the phrasing of rule 2. Keep every word of each: insert, do not rewrite, and do not copy the caption into them. Never add an object, a clause or a sentence that is not already there: no object the Self-reflect or Correction does not name, no "keeping ...", "while ... as described", "indeed", "remains" or similar additions, and no fact stated as satisfied. A fact about an object they do not name, or that only relates such an object to one they name, is left out of them. So is a fact about the very property they say is wrong or change (the colour of a teapot the Self-reflect calls silver instead of gold, the hair the Correction recolours): never write "the gold teapot is silver instead of gold". When no fact fits, return them unchanged.

Respond with one JSON object and nothing else:
{"caption": "...", "short_caption": "...", "self_reflect": "...", "correction": "...", "used": ["F1", "F3"]}
Leave "short_caption" as "" when no short caption is given, and "self_reflect" and "correction" as "" when no Self-reflect and Correction are given. "used" lists the ids of the facts that appear in the caption."""

WRITE_FEW_SHOT = [
    (
        {
            "caption": "A pride of lions rests under the shade of an acacia tree in a snowy mountain landscape, with snow covering the tree and the ground, and falling snowflakes visible in the air.",
            "short_caption": None,
            "self_reflect": "The scene is set in a savannah with golden grasses and a clear blue sky instead of a snowy mountain landscape with snow-covered ground and falling snowflakes.",
            "correction": "Replace the savannah with a snowy mountain landscape, add snow to the acacia tree and the ground, and include falling snowflakes.",
            "facts": [
                ("F1", "count", "There are exactly 7 lions."),
                ("F2", "position_3d", "The lions are in front of the tree."),
                ("F3", "negation", "There is no zebra in the image."),
            ],
        },
        {
            "caption": "A pride of seven lions rests under the shade of an acacia tree, lying in front of its trunk in a snowy mountain landscape, with snow covering the tree and the ground, falling snowflakes visible in the air, and no zebras in sight.",
            "short_caption": "",
            "self_reflect": "The scene is set in a savannah with golden grasses and a clear blue sky instead of a snowy mountain landscape with snow-covered ground and falling snowflakes.",
            "correction": "Replace the savannah with a snowy mountain landscape, add snow to the acacia tree and the ground, and include falling snowflakes.",
            "used": ["F1", "F2", "F3"],
        },
    ),
    (
        {
            "caption": "Two cups of coffee sit on a wooden table next to a small potted plant, with a chalkboard sign in the background.",
            "short_caption": "Two coffee cups and a plant on a wooden table.",
            "self_reflect": "The sign is missing.",
            "correction": "Add a chalkboard sign in the background.",
            "facts": [
                ("F1", "attribute", "The cup on the left is white and the cup on the right is blue."),
                ("F2", "position_2d", "The potted plant is to the right of the cups."),
                ("F3", "attribute", "The sign is wooden."),
                ("F4", "count", "There are exactly 3 cups."),
            ],
        },
        {
            "caption": "Two cups of coffee, a white one on the left and a blue one on the right, sit on a wooden table next to a small potted plant to the right of the cups, with a wooden chalkboard sign in the background.",
            "short_caption": "A white coffee cup and a blue coffee cup on a wooden table, with a potted plant to the right of the cups.",
            "self_reflect": "The sign is missing.",
            "correction": "Add a wooden chalkboard sign in the background.",
            "used": ["F1", "F2", "F3"],
        },
    ),
    (
        {
            "caption": "A woman in a beige cardigan sits in an armchair knitting a colorful scarf, in a cozy room with plants and a clock on the wall.",
            "short_caption": None,
            "self_reflect": "The woman has short gray hair instead of long dark hair.",
            "correction": "Change the woman's hair to long and dark.",
            "facts": [
                ("F1", "count", "There are exactly 4 flower pots."),
                ("F2", "position_2d", "The clock is to the left of the woman."),
                ("F3", "attribute", "The armchair is green."),
            ],
        },
        {
            "caption": "A woman in a beige cardigan sits in a green armchair knitting a colorful scarf, in a cozy room with plants in four flower pots and a clock on the wall to the left of the woman.",
            "short_caption": "",
            "self_reflect": "The woman has short gray hair instead of long dark hair.",
            "correction": "Change the woman's hair to long and dark.",
            "used": ["F1", "F2", "F3"],
        },
    ),
]


def _write_user_message(caption, short_caption, self_reflect, correction, facts) -> str:
    parts = [f"Caption: {caption}"]
    if short_caption:
        parts.append(f"Short caption: {short_caption}")
    elif short_caption == "":
        parts.append("Short caption: (none: write one)")
    if self_reflect:
        parts.append(f"Self-reflect: {self_reflect}")
    if correction:
        parts.append(f"Correction: {correction}")
    parts.append("Facts:")
    parts.extend(f"{fact_id} [{kind}] {text}" for fact_id, kind, text in facts)
    return "\n".join(parts)


def prompt_version(args) -> str:
    """Hash of everything that changes the output, so a prompt or threshold edit invalidates the cache."""
    payload = json.dumps(
        [VOCAB_PROMPT, REGION_PROMPT, sorted(STUFF | BODY_AND_WORN | SURFACES | UNIQUE_OBJECTS), WRITE_SYSTEM_PROMPT, WRITE_FEW_SHOT, FACT_PRIORITY, FACT_LEVEL, COLORS, MATERIALS, PATTERNS,
         VETO_PROMPT, args.veto, args.veto_counts, args.synth_short, args.short_max_words, args.min_box_frac, args.enrich_critique,
         args.max_position_facts, args.max_3d_facts,
         [os.path.basename(p.rstrip("/")) for p in (args.model_path, args.detector_path, args.depth_path, args.attribute_path,
                                                    args.sam3_path)],
         args.sam3_low, args.sam3_high,
         args.temperature, args.top_p, args.top_k, args.presence_penalty, args.vlm_max_pixels,
         args.det_low, args.det_high, args.attr_min_prob, args.attr_min_prob_material, args.attr_min_prob_pattern, args.depth_margin,
         args.max_count, args.max_regions, args.region_scene_pixels,
         args.region_crop_pixels, args.max_facts, args.negation_prob, args.max_dropped_words, args.min_used_fraction,
         args.seed, RULES_VERSION],
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------- row parsing

def human_caption(turns) -> str | None:
    """The first human turn when it is a plain caption (no image, no instruction), else None."""
    for turn in turns:
        if turn.get("from") == "human":
            value = turn.get("value") or ""
            if IMAGE_TOKEN in value or "orrection" in value:
                return None
            value = " ".join(value.split())
            return value or None
    return None


def critique_of(turns) -> tuple[str | None, str | None]:
    found = find_edit_turn(turns)
    if found is None:
        return None, None
    return found[1]["self_reflect"], found[1]["correction"]


def row_key(version: str, row_id, conversations, short_conversations) -> str:
    raw = json.dumps([version, row_id, conversations, short_conversations], ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()


def decode_image(item) -> Image.Image:
    if isinstance(item, dict):
        if item.get("bytes"):
            return Image.open(io.BytesIO(item["bytes"])).convert("RGB")
        return Image.open(item["path"]).convert("RGB")
    return Image.open(io.BytesIO(item)).convert("RGB")


def vlm_view(image: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale for the VLM; the card's processor default would otherwise spend thousands of tokens per image."""
    scale = (max_pixels / (image.width * image.height)) ** 0.5
    if scale >= 1:
        return image
    return image.resize((max(32, round(image.width * scale)), max(32, round(image.height * scale))), Image.BICUBIC)


def _json_object(text: str):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


# ----------------------------------------------------------------------------------------------------- the VLM

def vlm_generate(model, processor, batch_messages, batch_images, args, no_think_kwargs, max_tokens) -> list[str]:
    # The processor's template expects typed content parts.
    batch_messages = [[dict(m, content=[{"type": "text", "text": m["content"]}]) if isinstance(m["content"], str) else m
                       for m in messages] for messages in batch_messages]
    prompts = processor.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True, **no_think_kwargs)
    images = [image for row in batch_images for image in row]
    inputs = processor(text=prompts, images=images or None, padding=True, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    device = model.get_input_embeddings().weight.device
    inputs = {name: value.to(device) for name, value in inputs.items()}
    sampling = {}
    if args.temperature > 0:
        sampling = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "min_p": args.min_p}
    processors = [PresencePenalty(args.presence_penalty, input_length)] if args.presence_penalty else []
    with torch.inference_mode():
        output_ids = model.generate(**inputs, do_sample=args.temperature > 0, repetition_penalty=args.repetition_penalty,
                                    max_new_tokens=max_tokens, logits_processor=processors, **sampling)
    return processor.batch_decode(output_ids[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)


class VLM:
    def __init__(self, args):
        from transformers import AutoProcessor

        self.args = args
        self.model, _ = load_model(args)
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        # Batched generation with a decoder-only model needs left padding.
        self.processor.tokenizer.padding_side = "left"
        self.no_think = _resolve_no_think_kwargs(self.processor.tokenizer)
        self.batch_size = args.batch_size

    def run(self, jobs: list[dict], build, parse, max_tokens: int, stage: str) -> dict[int, tuple]:
        """{job index: (parsed or None, raw reply, attempts)}; replies that fail `parse` are retried."""
        results, attempts = {}, {index: 0 for index in range(len(jobs))}
        pending = list(range(len(jobs)))
        while pending:
            batch, pending = pending[:self.batch_size], pending[self.batch_size:]
            built = [build(jobs[index]) for index in batch]
            try:
                replies = vlm_generate(self.model, self.processor, [m for m, _ in built], [i for _, i in built],
                                       self.args, self.no_think, max_tokens)
            except torch.OutOfMemoryError:
                if self.batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                self.batch_size = max(1, self.batch_size // 2)
                logger.warning("%s: CUDA OOM, batch size -> %d", stage, self.batch_size)
                pending = batch + pending
                continue
            for index, reply in zip(batch, replies):
                attempts[index] += 1
                parsed = parse(reply, jobs[index])
                if parsed is None and attempts[index] <= self.args.max_retries:
                    pending.append(index)
                    continue
                results[index] = (parsed, reply, attempts[index])
        torch.cuda.empty_cache()  # the perception models run next on the same GPU
        return results


# ------------------------------------------------------------------------------------------ stage A: vocabulary

_NOUN_RE = re.compile(r"^[a-z][a-z' -]{0,40}$")


def build_vocab_messages(job, args):
    content = [{"type": "image"}, {"type": "text", "text": VOCAB_PROMPT.format(caption=job["caption"])}]
    return [{"role": "user", "content": content}], [vlm_view(job["final"], args.vlm_max_pixels)]


KINDS = ("person", "animal", "vehicle", "object")


def parse_vocab(reply: str, job) -> dict | None:
    value = _json_object(reply)
    if value is None or not isinstance(value.get("objects"), list):
        return None
    objects, seen = [], set()
    for item in value["objects"][:10]:
        if isinstance(item, str):
            item = {"name": item, "plural": item + "s"}
        if not isinstance(item, dict):
            return None
        name = " ".join(str(item.get("name", "")).lower().split())
        plural = " ".join(str(item.get("plural", "")).lower().split())
        # The VLM's plural only when it is the regular one or a known irregular one ("saucer" -> "sauces" happens).
        if plural not in (_pluralize(name), name + "s", name + "es") and plural not in _irregular_plurals(name):
            plural = _pluralize(name)
        # Garments, body parts and surfaces are not objects to place ("the tie is on the shirt", "the sign is behind
        # the street"); garments and hair are described per person in the region stage instead.
        if not _NOUN_RE.match(name) or name in seen or set(name.split()) & (STUFF | BODY_AND_WORN | SURFACES):
            continue
        seen.add(name)
        kind = str(item.get("kind", "object")).lower()
        objects.append({"name": name, "plural": plural, "kind": kind if kind in KINDS else "object"})
    absent = []
    for name in value.get("absent") or []:
        name = " ".join(str(name).lower().split())
        if _NOUN_RE.match(name) and name not in seen and name not in job["caption"].lower():
            absent.append(name)
    return {"objects": objects, "absent": absent[:3]}


# ------------------------------------------------------------------------------------------ stage B: perception

def _iou_matrix(boxes: torch.Tensor) -> torch.Tensor:
    from torchvision.ops import box_iou
    return box_iou(boxes, boxes)


def _contains(outer, inner, min_ioa=0.8) -> bool:
    x0, y0 = max(outer[0], inner[0]), max(outer[1], inner[1])
    x1, y1 = min(outer[2], inner[2]), min(outer[3], inner[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area >= min_ioa


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def clean_boxes(boxes: list[list[float]], scores: list[float], iou=0.5) -> list[int]:
    """NMS, then drop boxes that contain two or more other boxes (group boxes), then boxes lying inside a larger
    kept box of the same query (the sun's bright core inside its halo, the top half of a tower inside the whole
    tower: one object, two boxes, and "two suns" in the caption). Returns kept indices."""
    from torchvision.ops import nms

    if not boxes:
        return []
    keep = nms(torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32), iou).tolist()
    keep = [i for i in keep if sum(_contains(boxes[i], boxes[j]) for j in keep if j != i) < 2]
    return [i for i in keep if not any(_contains(boxes[j], boxes[i], 0.85) and _area(boxes[j]) > _area(boxes[i])
                                       for j in keep if j != i)]


class Perception:
    def __init__(self, args, device="cuda"):
        from transformers import (AutoImageProcessor, AutoModel, AutoModelForDepthEstimation,
                                  AutoModelForZeroShotObjectDetection, AutoProcessor, Sam3Model, Sam3Processor)

        self.args, self.device = args, device
        self.det_processor = AutoProcessor.from_pretrained(args.detector_path)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(args.detector_path).to(device).eval()
        self.depth_processor = AutoImageProcessor.from_pretrained(args.depth_path)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(args.depth_path).to(device).eval()
        self.sam3_processor = Sam3Processor.from_pretrained(args.sam3_path)
        self.sam3 = Sam3Model.from_pretrained(args.sam3_path, dtype=torch.bfloat16).to(device).eval()
        self.attr_processor = AutoProcessor.from_pretrained(args.attribute_path)
        self.attr_model = AutoModel.from_pretrained(args.attribute_path, dtype=torch.bfloat16).to(device).eval()
        self._text_features = {}

    # --- detection

    def detect(self, images: list[Image.Image], names: list[str]) -> list[dict[str, list[dict]]]:
        """Grounding DINO, one query per name (its text queries attend to each other, which hurts counting)."""
        return self.detect_pairs(images, [(i, name) for i in range(len(images)) for name in names])

    def det_inputs(self, images: list[Image.Image], texts: list[list[str]]):
        return self.det_processor(images=images, text=texts, return_tensors="pt", padding=True)

    @torch.inference_mode()
    def detect_pairs(self, images: list[Image.Image], pairs: list[tuple[int, str]]) -> list[dict[str, list[dict]]]:
        """detect() over arbitrary (image index, name) pairs, so images with different names share batches."""
        found = [{} for _ in images]
        for i, name in pairs:
            found[i][name] = []
        start, batch_size = 0, self.args.det_batch_size
        while start < len(pairs):
            chunk = pairs[start:start + batch_size]
            inputs = self.det_inputs([images[i] for i, _ in chunk], [[name] for _, name in chunk]).to(self.device)
            try:
                with torch.autocast("cuda", dtype=torch.float16):  # scores within 0.002 of fp32, 2.5x faster
                    outputs = self.detector(**inputs)
            except torch.OutOfMemoryError:  # the VLM's cached activations leave little room after a long batch
                if batch_size == 1:
                    raise
                del inputs
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                self.args.det_batch_size = batch_size  # later calls start from the size that fit
                logger.warning("detect: CUDA OOM, batch size -> %d", batch_size)
                continue
            start += len(chunk)
            results = self.det_processor.post_process_grounded_object_detection(
                outputs, threshold=self.args.det_low, text_threshold=self.args.det_low,
                target_sizes=[images[i].size[::-1] for i, _ in chunk])
            for (i, name), result in zip(chunk, results):
                boxes = result["boxes"].tolist()
                scores = result["scores"].tolist()
                labels = result.get("text_labels", result.get("labels"))
                picked = [k for k, label in enumerate(labels) if label]
                boxes, scores = [boxes[k] for k in picked], [scores[k] for k in picked]
                for k in clean_boxes(boxes, scores):
                    found[i][name].append({"box": [round(v, 1) for v in boxes[k]], "score": round(scores[k], 3)})
        return found

    # --- depth

    @torch.inference_mode()
    def depth(self, image: Image.Image) -> np.ndarray:
        """Relative inverse depth (larger = closer), normalised to [0, 1] with the 2nd/98th percentiles."""
        inputs = self.depth_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.depth_model(**inputs)
        depth = self.depth_processor.post_process_depth_estimation(outputs, target_sizes=[image.size[::-1]])[0]
        return self.normalize_depth(depth["predicted_depth"])

    @staticmethod
    def normalize_depth(predicted: torch.Tensor) -> np.ndarray:
        depth = predicted.float().cpu().numpy()
        low, high = np.percentile(depth, 2), np.percentile(depth, 98)
        return np.clip((depth - low) / max(high - low, 1e-6), 0, 1)

    # --- attributes

    def _embed(self, features):
        # Transformers 5.x returns a model output from get_*_features instead of a tensor.
        features = getattr(features, "pooler_output", features)
        return torch.nn.functional.normalize(features.float(), dim=-1)

    @torch.inference_mode()
    def _prompt_features(self, prompts: tuple[str, ...]):
        if prompts not in self._text_features:
            inputs = self.attr_processor(text=list(prompts), padding="max_length", max_length=64, return_tensors="pt").to(self.device)
            self._text_features[prompts] = self._embed(self.attr_model.get_text_features(**inputs))
        return self._text_features[prompts]

    @torch.inference_mode()
    def classify(self, image: Image.Image, boxes: list, masks: list, heads: dict[str, list[tuple[str, str]]]) -> list[dict]:
        """Zero-shot SigLIP over each masked region, pasted once on white and once on black: box crops let the
        background decide for small or thin objects (a grey robotic bee over lavender read as purple on the box and
        on the inner-box crop). heads = {key: [(value, prompt)]}; returns per region {key: [(value, p)]} sorted by p,
        where p is the smaller of the two backgrounds' softmax probabilities (0 for a mask covering <15% of its box)."""
        if not boxes:
            return []
        pixels = np.asarray(image)
        crops, usable = [], []
        for box, mask in zip(boxes, masks):
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(image.width, max(x0 + 1, x1)), min(image.height, max(y0 + 1, y1))
            region, keep = pixels[y0:y1, x0:x1], mask[y0:y1, x0:x1]
            usable.append(bool(keep.size) and keep.mean() >= 0.15)
            for background in (255, 0):
                crop = np.full_like(region, background)
                crop[keep] = region[keep]
                crops.append(Image.fromarray(crop))
        inputs = self.attr_processor(images=crops, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        image_features = self._embed(self.attr_model.get_image_features(**inputs))
        out = [{} for _ in boxes]
        for key, options in heads.items():
            text_features = self._prompt_features(tuple(prompt for _, prompt in options))
            probs = (image_features @ text_features.T * self.attr_model.logit_scale.exp().float()).softmax(-1)
            probs = torch.minimum(probs[0::2], probs[1::2]).tolist()
            for k in range(len(boxes)):
                ranked = sorted(((value, round(p if usable[k] else 0.0, 3)) for (value, _), p in zip(options, probs[k])),
                                key=lambda vp: -vp[1])
                out[k][key] = ranked
        return out

    def attributes(self, image: Image.Image, noun: str, instances: list[dict], masks: list[np.ndarray]) -> None:
        """instance[key] = [top value, p] and instance["probs"][key] = the ranked values, for color/material/pattern."""
        heads = {key: [(value, f"a photo of a {value} {noun}.") for value in vocab]
                 for key, vocab in (("color", COLORS), ("material", MATERIALS), ("pattern", PATTERNS))}
        for instance, ranked in zip(instances, self.classify(image, [i["box"] for i in instances], masks, heads)):
            instance["probs"] = ranked
            for key, values in ranked.items():
                instance[key] = list(values[0])

    # --- one image

    @torch.inference_mode()
    def detect_sam3(self, image: Image.Image, names: list[str]) -> dict[str, list[dict]]:
        """SAM 3 concept segmentation, one noun phrase at a time: instances with box, score and mask, cleaned like
        the Grounding DINO boxes. Its scores are bimodal (real instances mostly >= 0.9, junk <= 0.3), so the
        [sam3_low, sam3_high) band is small and tells when it is unsure (stacked pancakes at 0.36)."""
        out = {}
        for name in names:
            inputs = self.sam3_processor(images=image, text=name, return_tensors="pt").to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            outputs = self.sam3(**inputs)
            result = self.sam3_processor.post_process_instance_segmentation(
                outputs, threshold=self.args.sam3_low, mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist())[0]
            boxes = result["boxes"].float().cpu().tolist()
            scores = result["scores"].float().cpu().tolist()
            masks = result["masks"].cpu().numpy().astype(bool) if len(boxes) else []
            out[name] = [{"box": [round(v, 1) for v in boxes[k]], "score": round(scores[k], 3), "_mask": masks[k]}
                         for k in clean_boxes(boxes, scores)]
        return out

    def _instances(self, found: list[dict]) -> tuple[list[dict], int]:
        """Strong boxes, and the number of borderline boxes that are not just a duplicate or part of a strong one
        (only those make the count ambiguous)."""
        strong = [d for d in found if d["score"] >= self.args.det_high]
        isolated = 0
        for d in found:
            if d["score"] < self.args.det_high and not any(
                    _contains(s["box"], d["box"], 0.5) or _contains(d["box"], s["box"], 0.5) for s in strong):
                isolated += 1
        return strong, isolated

    def prepare_chunk(self, jobs: list[dict]) -> None:
        """Called once per chunk before analyze(); a subclass can batch the model calls here."""

    def analyze(self, image: Image.Image, vocab: dict) -> dict:
        """Scene measurements of one image: SAM 3 and Grounding DINO counts, masks, depth and attributes per noun,
        and which negation candidates SAM 3 finds no instance of."""
        names = [item["name"] for item in vocab["objects"]]
        found = self.detect([image], names)[0] if names else {}
        sam3 = self.detect_sam3(image, names + vocab["absent"]) if names or vocab["absent"] else {}
        depth = self.depth(image) if names else None
        min_area = self.args.min_box_frac * image.width * image.height

        def tidy(name, instances, gray):
            if name.split()[-1] in UNIQUE_OBJECTS and instances:  # one sun, however its halo gets boxed
                instances, gray = [max(instances, key=lambda i: i["score"])], 0
            # Tiny boxes are not counted (the test prompts count salient objects), but they hint that there are
            # more of the object than the count says, so they make the count unreliable.
            tiny = [i for i in instances if _area(i["box"]) < min_area]
            if tiny:
                instances, gray = [i for i in instances if _area(i["box"]) >= min_area], gray + len(tiny)
            return {"instances": instances, "gray": gray}

        # Two independent detector sides, each with its own one-box-one-noun assignment.
        gdino_side, sam3_side = {}, {}
        for item in vocab["objects"]:
            name = item["name"]
            gdino_side[name] = tidy(name, *self._instances(found[name]))
            strong = [i for i in sam3[name] if i["score"] >= self.args.sam3_high]
            sam3_side[name] = tidy(name, strong, len(sam3[name]) - len(strong))
        _assign_boxes(gdino_side)
        _assign_boxes(sam3_side)
        objects = {}
        for item in vocab["objects"]:
            name = item["name"]
            entry = {"gdino": gdino_side[name], "sam3": sam3_side[name],
                     "gdino_faint": [d["box"] for d in found[name] if d["score"] < self.args.det_high]}
            entry.update(count_vote(entry, self.args))
            objects[name] = entry
        # Depth from the object's own pixels: a shelf's mask minus the pots standing on it.
        everything = [i for e in objects.values() for i in e["instances"]]
        for instance in everything:
            mask = instance["_mask"].copy()
            for other in everything:
                if other is not instance and _contains(instance["box"], other["box"], 0.8):
                    mask &= ~other["_mask"]
            values = depth[mask] if mask.any() else depth[instance["_mask"]] if instance["_mask"].any() else None
            # The nearest surface (a high percentile of inverse depth) rather than the centre, as SpatialRGPT does.
            instance["depth"] = round(float(np.percentile(values, 80)), 3) if values is not None and values.size else None
        for item in vocab["objects"]:
            entry = objects[item["name"]]
            if entry["instances"]:
                self.attributes(image, item["name"], entry["instances"], [i["_mask"] for i in entry["instances"]])
        # A negation candidate the VLM proposed is absent when SAM 3 has no instance of it at all (>= sam3_low); the
        # veto stage checks the fact on the image again.
        absent = {name: not sam3[name] for name in vocab["absent"]}
        return {"size": list(image.size), "objects": objects, "absent": absent, "regions": [], "depth": depth}


# Words that name surfaces, regions or scene types, not countable things (after SpatialRGPT's remove_classes).
STUFF = set("""wall floor ceiling sky ground grass snow water sea ocean river lake field road street sidewalk
background foreground room kitchen bedroom scene view landscape area light shadow sunlight cloud clouds fog mist
forest mountain mountains hill hills beach sand dirt soil air smoke fire image photo picture painting art
""".split()) - {"picture", "painting"}


# Body parts and clothing are not objects to count or place ("the tie is on the shirt").
BODY_AND_WORN = set("""human face head hair eye eyes nose mouth ear hand hands arm leg foot beard body person-part
clothing shirt jacket coat trousers pants jeans shorts dress skirt sweater cardigan hat cap helmet glasses sunglasses
goggles footwear shoe shoes boot boots sock socks belt necklace bracelet earring earrings ring watch tie scarf glove
gloves bag backpack handbag vest suit t-shirt tshirt blouse apron headphone headphones hoodie uniform robe gown kimono
costume armor cloak sneaker sneakers sandal sandals bow-tie bowtie""".split())


# Surfaces the VLM tends to list as objects.
SURFACES = set("street sidewalk pavement road path lawn ceiling countertop tabletop".split())

# Things there is one of: their count is never above one, and nothing is "in front of" or "on" them.
UNIQUE_OBJECTS = set("sun moon sky earth horizon rainbow sunset sunrise".split())

# Buildings and rooms: countable and placeable, but nothing is "on" them (a ceiling lamp is not "on the house"), and
# they get no region description.
STRUCTURES = set("house building cottage cabin room structure barn castle church tower skyscraper hut tent shed".split())


def _assign_boxes(objects: dict, iou_threshold: float = 0.6) -> None:
    """One box, one noun. When two nouns claim the same region ("book" and "newspaper" both boxing the stack of
    newspapers), the higher-scoring noun keeps the instance; the other loses it and its count becomes unreliable
    (one more borderline box), so "two books" never reaches the caption."""
    names = list(objects)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for x in list(objects[a]["instances"]):
                for y in list(objects[b]["instances"]):
                    if (x in objects[a]["instances"] and y in objects[b]["instances"]
                            and _iou(x["box"], y["box"]) > iou_threshold):
                        loser, entry = (x, objects[a]) if x["score"] < y["score"] else (y, objects[b])
                        entry["instances"].remove(loser)
                        entry["gray"] += 1


_IRREGULAR = {"man": "men", "woman": "women", "child": "children", "person": "people", "mouse": "mice", "goose": "geese",
              "foot": "feet", "tooth": "teeth", "sheep": "sheep", "fish": "fish", "deer": "deer", "knife": "knives",
              "leaf": "leaves", "shelf": "shelves", "loaf": "loaves", "wolf": "wolves", "calf": "calves", "life": "lives",
              "wife": "wives", "half": "halves", "scarf": "scarves", "ox": "oxen", "cactus": "cacti"}


def _irregular_plurals(noun: str) -> set[str]:
    head, _, last = noun.rpartition(" ")
    for singular, plural in _IRREGULAR.items():
        if last.endswith(singular):  # "fireman", "bread knife"
            return {f"{head} {last[:-len(singular)]}{plural}".strip()}
    return set()


def _pluralize(noun: str) -> str:
    head, _, last = noun.rpartition(" ")
    if last.endswith(("s", "x", "ch", "sh")):
        last += "es"
    elif last.endswith("y") and last[-2:-1] not in "aeiou":
        last = last[:-1] + "ies"
    else:
        last += "s"
    return f"{head} {last}".strip()


# ------------------------------------------------------------------------------------------- stage C: regions

REGION_PROMPT = """The first image is a whole scene. Each image after it is a numbered crop from that scene showing one object, named before the crop. Describe every crop from what is visible in that crop only.

Return one JSON object and nothing else:
{"regions": [{"id": <crop number>, "color": "<color or null>", "material": "<material or null>"}, ...]}

- color: the dominant color of the object itself, not of its background. material: what it is made of (wooden, metal, glass, plastic, stone, ceramic, fabric, leather, paper), null for living things or when you cannot tell.
Leave out any field you would fill with null, and do not guess."""

_COLOR_WORDS = {"grey": "gray", "golden": "gold", "tan": "beige", "cream": "beige", "violet": "purple",
                "navy": "blue", "maroon": "red", "burgundy": "red", "crimson": "red", "turquoise": "blue",
                "teal": "green", "olive": "green", "ivory": "white", "khaki": "beige", "bronze": "brown",
                "copper": "orange", "magenta": "pink", "lavender": "purple"}
_MATERIAL_WORDS = {"wood": "wooden", "wooden": "wooden", "metal": "metal", "metallic": "metal", "steel": "metal",
                   "iron": "metal", "brass": "metal", "glass": "glass", "plastic": "plastic", "stone": "stone",
                   "marble": "stone", "ceramic": "ceramic", "porcelain": "ceramic", "clay": "ceramic",
                   "fabric": "fabric", "cloth": "fabric", "wool": "fabric", "cotton": "fabric", "leather": "leather",
                   "paper": "paper", "cardboard": "paper"}


def _normalize(value, table: dict, allowed) -> str | None:
    """Last word of a free-form attribute that maps onto the vocabulary ("dark brown" -> "brown")."""
    if not isinstance(value, str):
        return None
    for word in reversed(re.findall(r"[a-z]+", value.lower())):
        word = table.get(word, word)
        if word in allowed:
            return word
    return None


def count_vote(entry: dict, args) -> dict:
    """Count of one noun from the two counters only (VLMs miscount): SAM 3 (primary) and Grounding DINO. SAM 3's
    instances are always the objects (they carry the masks). Rules:
      - the two must agree on the value
      - if as many as half of SAM 3's instances are tiny/borderline the count is unreliable (a forest, not "five
        trees"); one borderline instance next to two or more clean ones (a reflection, a half-hidden curtain) is
        settled by the two agreeing exactly, more are not, and a lone box with a borderline twin stays unreliable
        (one cup or two?)
      - SAM 3 is never overruled: with Florence-2 gone there is no third vote, and Grounding DINO alone over SAM 3
        counted a far-off figure as a third gondolier
      - one clean SAM 3 instance that Grounding DINO only sees faintly (a borderline box on it) is one object for
        geometric and region facts; counts of one never become count facts
    Returns {count, boxes_match, instances, agree, n_votes}; boxes_match says `instances` are the counted objects."""
    g, s = entry["gdino"], entry["sam3"]
    n_s, n_g = len(s["instances"]), len(g["instances"])

    def out(count):
        return {"count": count, "boxes_match": count is not None, "instances": s["instances"],
                "agree": 2 if n_s == n_g else 1, "n_votes": 2}

    if n_s == n_g and 1 <= n_s <= args.max_count:
        if s["gray"] >= max(2, n_s // 2) or (s["gray"] and n_s < 2):
            return out(None)
        return out(n_s)
    if (n_s == 1 and not s["gray"] and n_g == 0
            and any(_iou(box, s["instances"][0]["box"]) >= 0.5 for box in entry.get("gdino_faint", []))):
        return out(1)
    return out(None)


def select_regions(scene: dict, vocab: dict, counts: dict, args) -> list[dict]:
    """Singletons, and both members of a pair told apart by left/right, largest first: the instances a caption can
    refer to unambiguously ("the chair", "the cup on the left")."""
    regions = []
    for item in vocab["objects"]:
        count, boxes_match = counts[item["name"]]
        # People have no color or material to check (their clothes are not described).
        if not boxes_match or count not in (1, 2) or item["name"].split()[-1] in STRUCTURES or item["kind"] == "person":
            continue
        instances = sorted(scene["objects"][item["name"]]["instances"], key=lambda i: i["box"][0] + i["box"][2])
        if count == 2 and _centers(instances[1:])[0][0] - _centers(instances[:1])[0][0] < 0.1 * scene["size"][0]:
            continue
        for k, instance in enumerate(instances):
            ref = f"the {item['name']}" if count == 1 else f"the {item['name']} on the {'left' if k == 0 else 'right'}"
            box = instance["box"]
            regions.append({"name": item["name"], "kind": item["kind"], "ref": ref, "instance": instance,
                            "area": (box[2] - box[0]) * (box[3] - box[1])})
    regions.sort(key=lambda r: -r["area"])
    return regions[:args.max_regions]


def region_crop(image: Image.Image, box, pad=0.1) -> tuple[Image.Image, list[float]]:
    """Padded crop at full resolution and the object's box in crop coordinates."""
    x0, y0, x1, y1 = box
    dx, dy = pad * (x1 - x0), pad * (y1 - y0)
    left, top = max(0, x0 - dx), max(0, y0 - dy)
    crop = image.crop((int(left), int(top), int(min(image.width, x1 + dx)), int(min(image.height, y1 + dy))))
    return crop, [x0 - int(left), y0 - int(top), x1 - int(left), y1 - int(top)]


def prepare_regions(job: dict, rng: random.Random, args) -> None:
    """Crops for the VLM (shuffled, so the numbering carries no size order)."""
    views = []
    for index, region in enumerate(job["regions"]):
        region["crop"], region["crop_box"] = region_crop(job["final"], region["instance"]["box"])
        views.append({"region": index})
    rng.shuffle(views)
    for number, view in enumerate(views, 1):
        view["id"] = number
    job["region_views"] = views


def build_region_messages(job, args):
    content = [{"type": "image"}, {"type": "text", "text": REGION_PROMPT}]
    images = [vlm_view(job["final"], args.region_scene_pixels)]
    for view in job["region_views"]:
        region = job["regions"][view["region"]]
        content += [{"type": "text", "text": f"Crop {view['id']} ({region['name']}):"}, {"type": "image"}]
        images.append(vlm_view(region["crop"], args.region_crop_pixels))
    return [{"role": "user", "content": content}], images


def parse_regions(reply: str, job) -> dict | None:
    value = _json_object(reply)
    if value is None or not isinstance(value.get("regions"), list):
        return None
    answers = {}
    for entry in value["regions"]:
        if isinstance(entry, dict) and isinstance(entry.get("id"), int):
            answers[entry["id"]] = entry
    return answers if answers else None


def verify_region(region: dict, answer: dict) -> dict:
    """What survives the check, for one region: the VLM names a color / material, SigLIP checks it on the object's
    mask (it must be among SigLIP's two best). Neither alone is trusted: SigLIP alone called a weathered grey chair
    purple, and the VLM alone colors from the background."""
    verified = {"ref": region["ref"], "name": region["name"], "kind": region["kind"]}
    probs = region["instance"].get("probs") or {}

    def top2(key, value):
        ranked = probs.get(key) or []
        return value is not None and any(v == value and p >= 0.1 for v, p in ranked[:2])
    color = _normalize(answer.get("color"), _COLOR_WORDS, set(COLORS))
    if top2("color", color):
        verified["color"] = color
    if region["kind"] in ("object", "vehicle"):
        material = _normalize(answer.get("material"), _MATERIAL_WORDS, set(MATERIALS) - NEUTRAL)
        if top2("material", material):
            verified["material"] = material
    return verified


def describe_regions(jobs: list[dict], vlm, args) -> None:
    """Stage C for a chunk: one VLM call per image over its crops, then per-claim verification."""
    todo = [job for job in jobs if job.get("regions")]
    if not todo:
        return
    results = vlm.run(todo, lambda job: build_region_messages(job, args), parse_regions, args.region_max_new_tokens, "regions")
    for index, job in enumerate(todo):
        answers = results[index][0] or {}
        job["region_raw"] = {job["regions"][view["region"]]["ref"]: answers.get(view["id"]) for view in job["region_views"]}
        job["scene"]["regions"] = [verify_region(job["regions"][view["region"]], answers.get(view["id"]) or {})
                                   for view in sorted(job["region_views"], key=lambda v: v["region"])]


# ------------------------------------------------------------------------------------------------ stage C: facts

def _number(n: int) -> str:
    return NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else str(n)


def _group(name: str, plural: str, count: int) -> str:
    return f"the {name}" if count == 1 else f"the {plural}"


def _union(instances):
    boxes = np.array([i["box"] for i in instances])
    return boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()


def _centers(instances):
    return [((i["box"][0] + i["box"][2]) / 2, (i["box"][1] + i["box"][3]) / 2) for i in instances]


def _shared_attribute(instances, key, threshold) -> str | None:
    """The value every instance agrees on with enough confidence, unless it is a neutral one."""
    values = [instance.get(key) for instance in instances]
    if not values or any(v is None or v[1] < threshold for v in values) or len({v[0] for v in values}) != 1:
        return None
    return None if values[0][0] in NEUTRAL else values[0][0]


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _same_layer(a, b, margin) -> bool:
    depths = [i.get("depth") for i in a + b]
    if any(d is None for d in depths):
        return False
    return abs(float(np.median([i["depth"] for i in a])) - float(np.median([i["depth"] for i in b]))) <= margin


def relation_2d(a, b, size) -> str | None:
    """Relation of every instance of `a` to every instance of `b`, or None when the groups interleave."""
    width, height = size
    ca, cb = _centers(a), _centers(b)
    ua, ub = _union(a), _union(b)
    # One spans the other horizontally (a table under a man, a tree over lions): left/right would be noise.
    stacked = min(ua[2], ub[2]) - max(ua[0], ub[0]) >= 0.5 * min(ua[2] - ua[0], ub[2] - ub[0])
    if stacked:
        pass
    elif max(x for x, _ in ca) < min(x for x, _ in cb) - 0.03 * width:
        return "to the left of"
    elif min(x for x, _ in ca) > max(x for x, _ in cb) + 0.03 * width:
        return "to the right of"
    x_overlap = min(ua[2], ub[2]) - max(ua[0], ub[0])
    # Above/below need a visible gap: stacked objects that touch are "on", and the rewrite would turn "a chair on
    # the rug" into "a chair above the rug".
    if x_overlap > 0.3 * min(ua[2] - ua[0], ub[2] - ub[0]):
        if ua[3] < ub[1] - 0.02 * height:
            return "above"
        if ua[1] > ub[3] + 0.02 * height:
            return "below"
    return None


def relation_3d(a, b, margin) -> str | None:
    depths_a = [i["depth"] for i in a if i.get("depth") is not None]
    depths_b = [i["depth"] for i in b if i.get("depth") is not None]
    if len(depths_a) != len(a) or len(depths_b) != len(b):
        return None
    # Depth order only reads as front/behind when the objects line up (per instance: a journal between two pens at
    # the edges of the table is not "in front of the pens", although the pens' union box spans it), and not when
    # one box mostly lies in the other (the table standing on the rug is not "in front of the rug").
    def lined_up(x, y):
        return min(x[2], y[2]) - max(x[0], y[0]) >= 0.3 * min(x[2] - x[0], y[2] - y[0])
    if not any(lined_up(x["box"], y["box"]) for x in a for y in b):
        return None
    if any(_contains(x["box"], y["box"], 0.5) or _contains(y["box"], x["box"], 0.5) for x in a for y in b):
        return None
    if min(depths_a) > max(depths_b) + margin:
        return "in front of"
    if max(depths_a) < min(depths_b) - margin:
        return "behind"
    return None


def _same_object(a, b) -> bool:
    """Two vocabulary entries that the detector resolved to the same things (e.g. "person" and "woman")."""
    from torchvision.ops import box_iou
    iou = box_iou(torch.tensor([i["box"] for i in a]), torch.tensor([i["box"] for i in b]))
    return bool((iou > 0.7).any())


def _nested(a, b) -> bool:
    """Some instance of one kind lies inside an instance of the other: worn, held or part of it (a hat on a woman,
    a bicycle's basket), where left/right/above/below read as nonsense ("the women are below the hats")."""
    return any(_contains(x["box"], y["box"], 0.7) or _contains(y["box"], x["box"], 0.7) for x in a for y in b)


def _is_background(instances, size) -> bool:
    x0, y0, x1, y1 = _union(instances)
    return (x1 - x0) * (y1 - y0) > 0.6 * size[0] * size[1]


def _rests_on(top: dict, bottom: dict, height: int, depth: np.ndarray | None = None, margin: float = 0.08) -> bool:
    """The top object's mask sits on the bottom one's: for most columns of the top mask, the pixels just under its
    lowest point belong to the bottom mask (the support test SpatialRGPT defines, done on masks instead of points).
    The top object also has to stand within the bottom one's box (a wall clock above a sofa touches the sofa's
    mask in the image but hangs above the sofa's box) and the two surfaces have to meet in depth (the clock is on
    the wall, farther than the sofa's back)."""
    upper, lower = top["_mask"], bottom["_mask"]
    if upper.sum() == 0 or upper.sum() >= lower.sum():
        return False
    tb, bb = top["box"], bottom["box"]
    if not (bb[1] <= tb[3] <= bb[3] and bb[0] <= (tb[0] + tb[2]) / 2 <= bb[2]):
        return False
    cols = np.where(upper.any(axis=0))[0]
    lowest = upper.shape[0] - 1 - np.argmax(upper[::-1, cols], axis=0)
    band = max(3, int(0.02 * height))
    touching = np.zeros(len(cols), dtype=bool)
    for offset in range(1, band + 1):
        rows = np.minimum(lowest + offset, lower.shape[0] - 1)
        touching |= lower[rows, cols]
    if touching.mean() < 0.5:
        return False
    if depth is not None:
        above, below = [], []
        for k in np.where(touching)[0]:
            for offset in range(0, band):
                row = max(0, lowest[k] - offset)
                if upper[row, cols[k]]:
                    above.append(depth[row, cols[k]])
            for offset in range(1, band + 1):
                row = min(lowest[k] + offset, lower.shape[0] - 1)
                if lower[row, cols[k]]:
                    below.append(depth[row, cols[k]])
        if above and below and abs(float(np.median(above)) - float(np.median(below))) > margin:
            return False
    return True


def _restates(value: str, noun: str) -> bool:
    """The attribute is already the noun ("paper" for "paper", "wooden" for "wood carving", "glass" for "glass surface")."""
    words = set(noun.split())
    return value in words or value.rstrip("en") in words or any(w.startswith(value.rstrip("en")) for w in words)


def region_facts(regions: list[dict], add) -> None:
    for region in regions:
        ref, name = region["ref"], region["name"]
        Ref = ref[0].upper() + ref[1:]
        key = ref.replace(" ", "_")
        if region.get("color") and region["color"] not in name.split():
            add(f"color:{key}", "attribute", region["color"], f"{Ref} is {region['color']}.", [name])
        if region.get("material") and not _restates(region["material"], name):
            add(f"material:{key}", "attribute", region["material"], f"{Ref} is {region['material']}.", [name])


_VAGUE_QUANTITY = set("group pride herd flock pack crowd bunch number variety row set series collection few several couple lot lots stack pile".split())


def _caption_count_value(caption: str, name: str, plural: str) -> int | None:
    """The number the caption gives this object ("a book" -> 1, "two cups" -> 2, "3 dogs" -> 3), or None. Vague
    quantities ("a group of lions", "a stack of books") are not numbers."""
    numbers = "|".join(NUMBER_WORDS[1:] + ["a", "an", "single", "pair of", r"\d+"])
    nouns = "|".join(re.escape(n) for n in {name, plural})
    # A lookahead, so that a skipped match does not swallow the real one ("a scene with two people").
    for match in re.finditer(rf"(?i)(?=\b({numbers})\b((?:\s+[\w-]+){{0,3}}?)\s+({nouns})\b)", caption):
        between = set(re.findall(r"[a-z]+", match.group(2).lower()))
        if between & _VAGUE_QUANTITY:
            continue
        if "pair" in between:  # "a pair of pens"
            return 2
        word = match.group(1).lower()
        # "a park bench feeding pigeons" is not one pigeon: an article only counts its own singular noun.
        if word in ("a", "an", "single") and plural != name and match.group(3).lower() == plural.lower():
            continue
        if word.isdigit():
            return int(word)
        if word in NUMBER_WORDS:
            return NUMBER_WORDS.index(word)
        return 2 if word == "pair of" else 1
    return None


def _caption_counts(caption: str, name: str, plural: str) -> bool:
    """The caption already gives this object a number. The caption's count wins, so the fact is not offered at
    all: offered, the rewrite turns "reading a book" into "reading one of two books"."""
    return _caption_count_value(caption, name, plural) is not None


def derive_facts(scene: dict, vocab: dict, counts: dict, args, rng: random.Random, caption: str = "") -> dict[str, dict]:
    """{key: {"type", "value", "text", "objects"}} measured on one image. The key names the claim (which objects,
    which property), "value" is what was measured and "text" is the wording given to the LLM."""
    facts = {}
    plural = {item["name"]: item["plural"] for item in vocab["objects"]}
    kind = {item["name"]: item["kind"] for item in vocab["objects"]}
    # The caption is the reference for the final image: a detector count that contradicts a number the caption
    # states ("a book" against two boxes, the second being the stack of newspapers) means the boxes are not that
    # object, so the noun takes part in no fact at all, not just in no count fact.
    # Unless every detector that voted agrees on another number, and the two counters (SAM 3 and Grounding DINO)
    # have no borderline box: then the caption is wrong ("three women and one woman" over five women, "a fishing
    # rod" over two) and the count fact says so; the veto stage checks such facts on the image.
    counts = dict(counts)
    overrides = set()
    for name in plural:
        stated = _caption_count_value(caption, name, plural[name])
        if stated is not None and counts[name][0] is not None and stated != counts[name][0]:
            entry = scene["objects"][name]
            unanimous = entry.get("agree") == entry.get("n_votes")
            clean = not entry["sam3"]["gray"] and not entry["gdino"]["gray"]
            if unanimous and clean:
                overrides.add(name)
            else:
                counts[name] = (None, False)
    count_of = {name: counts[name][0] for name in plural}
    # Geometry needs the detector boxes to be the counted objects.
    reliable = sorted(name for name in plural if counts[name][0] and counts[name][1])
    described = {r["ref"] for r in scene.get("regions", [])}

    def add(key, kind_, value, text, objects, **extra):
        facts[key] = {"type": kind_, "value": value, "text": text, "objects": objects, **extra}

    for name in plural:
        # "a basket" -> "one basket" teaches nothing, and a count the caption states already wins over the fact.
        if count_of[name] and count_of[name] >= 2 and name in overrides:
            add(f"count:{name}", "count", count_of[name],
                f"The image shows {_number(count_of[name])} {plural[name]}; the caption's number for them is wrong, so write {_number(count_of[name])} {plural[name]}.",
                [name], override=True)
        elif count_of[name] and count_of[name] >= 2 and not _caption_counts(caption, name, plural[name]):
            add(f"count:{name}", "count", count_of[name], f"There are exactly {count_of[name]} {plural[name]}.", [name])

    for name in reliable:
        count, instances = count_of[name], scene["objects"][name]["instances"]
        # Groups: SigLIP alone, but every member must agree. Singletons and pairs get their attributes from the
        # region stage, where the VLM and SigLIP must agree (SigLIP alone called a weathered grey chair purple).
        if count >= 2 and kind[name] != "person" and f"the {name} on the left" not in described:
            heads = [("color", args.attr_min_prob)]
            if kind[name] in ("object", "vehicle"):
                heads += [("material", args.attr_min_prob_material), ("pattern", args.attr_min_prob_pattern)]
            for key, threshold in heads:
                value = _shared_attribute(instances, key, threshold)
                if value is not None and not _restates(value, name):
                    add(f"{key}:{name}", "attribute", value, f"All {count} {plural[name]} are {value}.", [name])
        if count == 2:
            left, right = sorted(instances, key=lambda i: i["box"][0] + i["box"][2])
            if (right["box"][0] + right["box"][2]) - (left["box"][0] + left["box"][2]) > 0.1 * scene["size"][0]:
                area = lambda i: (i["box"][2] - i["box"][0]) * (i["box"][3] - i["box"][1])
                if max(area(left), area(right)) >= args.size_ratio * min(area(left), area(right)):
                    bigger, smaller = ("left", "right") if area(left) > area(right) else ("right", "left")
                    add(f"size:{name}", "comparison", bigger,
                        f"The {name} on the {bigger} is larger than the {name} on the {smaller}.", [name])

    # Pairs pinned down by left/right keep distinct colors as one fact ("the car on the left is red").
    regions = [dict(r) for r in scene.get("regions", [])]
    by_ref = {r["ref"]: r for r in regions}
    for name in reliable:
        left, right = by_ref.get(f"the {name} on the left"), by_ref.get(f"the {name} on the right")
        if left and right and left.get("color") and right.get("color") and left["color"] != right["color"]:
            add(f"instance_color:{name}", "attribute", [left["color"], right["color"]],
                f"The {name} on the left is {left['color']} and the {name} on the right is {right['color']}.", [name])
            left.pop("color"), right.pop("color")
    region_facts(regions, add)

    height = scene["size"][1]
    sky = {name for name in reliable if name.split()[-1] in UNIQUE_OBJECTS}  # nothing is on or in front of the sun
    for i, a in enumerate(reliable):
        for b in reliable[i + 1:]:
            ia, ib = scene["objects"][a]["instances"], scene["objects"][b]["instances"]
            if _same_object(ia, ib):
                continue
            # "on": every instance of one rests on some instance of the other (bread slices on a board).
            for top, bottom, it, ib_ in ((a, b, ia, ib), (b, a, ib, ia)):
                # A held book is not "on the woman"; a ceiling lamp is not "on the house".
                if kind[bottom] in ("person", "animal") or top in sky or bottom in sky or bottom.split()[-1] in STRUCTURES:
                    continue
                if all(any(_rests_on(t, u, height, scene.get("depth"), args.depth_margin) for u in ib_) for t in it):
                    verb = "is" if count_of[top] == 1 else "are"
                    add(f"on:{top}:{bottom}", "support", True,
                        f"{_group(top, plural[top], count_of[top]).capitalize()} {verb} on {_group(bottom, plural[bottom], count_of[bottom])}.",
                        [top, bottom])
            if _nested(ia, ib) or f"on:{a}:{b}" in facts or f"on:{b}:{a}" in facts:
                continue
            # Measured as "a <relation> b" with a < b; the wording may put either object first.
            swap = rng.random() < 0.5
            def phrase(relation):
                (s_, o, rel) = (b, a, _INVERSE[relation]) if swap else (a, b, relation)
                verb = "is" if count_of[s_] == 1 else "are"
                return f"{_group(s_, plural[s_], count_of[s_]).capitalize()} {verb} {rel} {_group(o, plural[o], count_of[o])}."
            # A frame-filling object (the sunflower under a close-up ladybug) relates by "on"/"in", not by position.
            if not (_is_background(ia, scene["size"]) or _is_background(ib, scene["size"])):
                relation = relation_2d(ia, ib, scene["size"])
                # Above/below across depth layers reads as nonsense ("the trees are above the book").
                if relation in ("above", "below") and not _same_layer(ia, ib, args.depth_margin):
                    relation = None
                if relation:
                    add(f"2d:{a}:{b}", "position_2d", relation, phrase(relation), [a, b])
                relation = None if a in sky or b in sky else relation_3d(ia, ib, args.depth_margin)
                if relation:
                    add(f"3d:{a}:{b}", "position_3d", relation, phrase(relation), [a, b])
            if count_of[a] != count_of[b] and min(count_of[a], count_of[b]) >= 2:  # "more windows than doors" with one door reads oddly
                more, fewer = (a, b) if count_of[a] > count_of[b] else (b, a)
                add(f"more:{a}:{b}", "comparison", more, f"There are more {plural[more]} than {plural[fewer]}.", [a, b])

    for name, is_absent in scene["absent"].items():
        if is_absent:
            add(f"no:{name}", "negation", True, f"There is no {name} in the image.", [name])

    return facts


_INVERSE = {"to the left of": "to the right of", "to the right of": "to the left of", "above": "below",
            "below": "above", "in front of": "behind", "behind": "in front of"}


# Scarcest and most needed by the OOD sets first (count/position edits fail most there); negations are always
# available ("no zebra"), so they come last.
FACT_PRIORITY = ["count", "position", "attribute", "comparison", "negation"]
# Fact types that share a priority level.
FACT_LEVEL = {"position_2d": "position", "position_3d": "position", "support": "position"}


# Within the position level: what the test prompts ask for most first, and the noisiest measurement last.
_POSITION_RANK = {"position_2d": 0, "support": 1, "position_3d": 2}


def select_facts(candidates: list[dict], rng: random.Random, args, caption: str = "") -> list[dict]:
    """Up to max_facts facts in FACT_PRIORITY order, shuffled within a priority level. Negations are offered only in
    a fraction of rows and at most once per row: they would otherwise land in almost every caption, and "with no cars,
    benches, or dogs" reads as a checklist. Position facts are ranked, not shuffled: pairs the caption names first,
    left/right before on before in front of/behind, at most max_3d_facts depth facts and max_position_facts in all,
    so a still life does not turn into "behind the saucer, in front of the doily, with a curtain behind the cat and
    behind the table"."""
    by_level = {level: [] for level in FACT_PRIORITY}
    for fact in candidates:
        by_level[FACT_LEVEL.get(fact["type"], fact["type"])].append(fact)
    for facts in by_level.values():
        rng.shuffle(facts)
    text = caption.lower()
    mentioned = lambda fact: sum(re.search(rf"\b{re.escape(o)}s?\b", text) is not None for o in fact["objects"])
    positions = sorted(by_level["position"], key=lambda f: (-mentioned(f), _POSITION_RANK.get(f["type"], 9)))
    kept, depth_facts = [], 0
    for fact in positions:
        if fact["type"] == "position_3d":
            if depth_facts >= args.max_3d_facts:
                continue
            depth_facts += 1
        kept.append(fact)
    by_level["position"] = kept[:args.max_position_facts]
    by_level["negation"] = by_level["negation"][:1] if rng.random() < args.negation_prob else []
    return [fact for level in FACT_PRIORITY for fact in by_level[level]][:args.max_facts]


# ------------------------------------------------------------------------------------------ stage D: veto

VETO_PROMPT = """Look at the image and check each statement about it, from the image alone.

Statements:
{statements}

Return one JSON object and nothing else, one entry per statement id: {{"F1": "yes", "F2": "no", ...}}
- "yes": clearly true.
- "no": clearly false: the object named is not there or is something else, a relation is the other way round or does not hold (the thing is not on that surface, is not in front of or behind it), a color or material is not what is said.
- "unsure": you cannot tell from the image.
The counts were measured by detectors: do not answer "no" to a count because you would count differently, only when the objects named are not in the image. A count that says the caption's number is wrong contradicts the caption: answer "yes" only if that count is clearly right in the image."""


def build_veto_messages(job, args):
    statements = "\n".join(f"{fact['id']}: {fact['text']}" for fact in job["selected"])
    content = [{"type": "image"}, {"type": "text", "text": VETO_PROMPT.format(statements=statements)}]
    return [{"role": "user", "content": content}], [vlm_view(job["final"], args.vlm_max_pixels)]


def parse_veto(reply: str, job) -> dict | None:
    value = _json_object(reply)
    if value is None:
        return None
    answers = {}
    for fact in job["selected"]:
        answer = str(value.get(fact["id"], "unsure")).strip().lower()
        answers[fact["id"]] = answer if answer in ("yes", "no", "unsure") else "unsure"
    return answers


def veto_facts(jobs: list[dict], vlm, args) -> None:
    """The VLM may only remove facts, never add or change them: a measured fact answered "no" on the image is
    dropped (counts only with --veto_counts, since VLMs miscount). The remaining facts are renumbered."""
    todo = [job for job in jobs if job.get("selected")]
    if not todo:
        return
    results = vlm.run(todo, lambda job: build_veto_messages(job, args), parse_veto, args.veto_max_new_tokens, "veto")
    for index, job in enumerate(todo):
        answers = results[index][0] or {}
        job["vetoed"] = [fact for fact in job["selected"]
                         if answers.get(fact["id"]) == "no" and (fact["type"] != "count" or args.veto_counts or fact.get("override"))]
        job["veto_answers"] = {fact["key"]: answers.get(fact["id"], "unsure") for fact in job["selected"]}
        job["selected"] = [fact for fact in job["selected"] if fact not in job["vetoed"]][:args.max_facts]
        for number, fact in enumerate(job["selected"], 1):
            fact["id"] = f"F{number}"


# ------------------------------------------------------------------------------------------ stage E: rewriting

def build_write_messages(job, args):
    messages = [{"role": "system", "content": WRITE_SYSTEM_PROMPT.replace("{short_max_words}", str(args.short_max_words))}]
    for example_in, example_out in WRITE_FEW_SHOT:
        messages.append({"role": "user", "content": "(example, image omitted)\n" + _write_user_message(**example_in)})
        messages.append({"role": "assistant", "content": json.dumps(example_out, ensure_ascii=False)})
    facts = [(fact["id"], fact["type"], fact["text"]) for fact in job["selected"]]
    short = job["short_caption"] or ("" if args.synth_short else None)
    text = _write_user_message(job["caption"], short, job["self_reflect"], job["correction"], facts)
    messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]})
    return messages, [vlm_view(job["final"], args.vlm_max_pixels)]


_QUANTITY_WORDS = set("a an one some several many few group pride herd flock pack crowd bunch couple pair number various multiple".split()) | set(NUMBER_WORDS)
_AFTERTHOUGHT = re.compile(r"(?i)\b(totaling|totalling|a total of|in total|in number|specifically|namely|in particular|one of (?:the )?(?:two|three|four|five|six|seven|eight|nine|ten))\b")
# Prepositions are not here on purpose: "on the rug" -> "above the rug" changes the meaning.
_FUNCTION_WORDS = set("the and or of is are be its it that this as there their".split())
def _foreign_words(text: str, caption: str, facts: list[dict]) -> list[str]:
    """Content words of `text` that neither the caption nor the facts contain (stemmed): objects, colors or actions
    the rewrite invented."""
    allowed = {_stem(w) for w in _words(caption + " " + " ".join(f["text"] for f in facts))}
    return [w for w in _words(text)
            if _stem(w) not in allowed and w not in _FUNCTION_WORDS and w not in _QUANTITY_WORDS and w not in _PROMPT_GLUE]


def _words(text: str) -> list[str]:
    """Lowercase words; a quote around a word ('Friday Tales') is not part of it."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def _stem(word: str) -> str:
    """Crude stem so that "resting", "rests", "covered" and "covering" count as the caption's own words."""
    for suffix in ("ing", "ed", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[:-len(suffix)]
            break
    return word[:-1] if word.endswith("e") and len(word) > 3 else word


# Words a synthesized short caption may add without naming anything new.
_PROMPT_GLUE = set(NUMBER_WORDS) | set("""with without no in on at to left right front behind above below next beside under over facing wearing
has have by near between among along around across inside outside atop onto into from for while and or""".split())


_EXACTLY = re.compile(r"(?i)\bexactly\s+(?=(?:" + "|".join(NUMBER_WORDS) + r"|\d+)\b)")


def _strip_exactly(original: str, new: str) -> str:
    """"containing exactly two chairs" -> "containing two chairs" when the original never says "exactly"."""
    return new if "exactly" in original.lower() else _EXACTLY.sub("", new)


def _new_digits(original: str, new: str, facts: list[dict]) -> bool:
    allowed = set(re.findall(r"\d+", original)) | {d for f in facts if f["type"] == "text" for d in re.findall(r"\d+", f["string"])}
    return bool(set(re.findall(r"\d+", new)) - allowed)


def _spell_counts(original: str, new: str, facts: list[dict]) -> str:
    """"2 pillows" -> "two pillows" for a digit the rewrite added that is the value of a count or part fact."""
    original_digits = set(re.findall(r"\d+", original))
    for value in {f["value"] for f in facts if f["type"] in ("count", "part") and isinstance(f["value"], int)}:
        if str(value) not in original_digits:
            new = re.sub(rf"\b{value}\b(?=\s+[A-Za-z])", _number(value), new)
    return new


_RELATION_RE = re.compile(r"\b((?:to the (?:left|right) of|in front of|behind|above|below|on top of|beneath|next to|beside|underneath) (?:the |a |an |his |her |its |their )?[\w-]+)")


def _relation_phrases(text: str) -> list[str]:
    """Position phrases of the caption ("to the right of the sofa"): a rewrite may add relations, never lose one."""
    return [m.lower() for m in _RELATION_RE.findall(text)]


def _quoted(text: str) -> list[str]:
    return [a or b for a, b in re.findall(r"\"([^\"]+)\"|(?<![A-Za-z])'([^']+)'(?![A-Za-z])", text)]


def dropped_words(original: str, new: str) -> list[str]:
    """Words of the original that the rewrite lost (vague quantities, which a count may replace, excepted)."""
    kept = set(re.findall(r"[a-z0-9']+", new.lower()))
    return [w for w in re.findall(r"[a-z0-9']+", original.lower())
            if w not in kept and w not in _QUANTITY_WORDS and w not in _FUNCTION_WORDS and not w.isdigit()]


def _clean_caption(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    if IMAGE_TOKEN in value or re.search(r"(?i)\b(self-reflect|correction)\s*:", value):
        return None
    return value


def make_write_parser(args):
    def parse(reply: str, job) -> dict | None:
        value = _json_object(reply)
        if value is None:
            return None
        caption = _clean_caption(value.get("caption"))
        if not caption or coverage(job["caption"], [caption]) < args.min_coverage:
            return None
        if len(caption.split()) > len(job["caption"].split()) + args.max_added_words:
            return None
        if len(dropped_words(job["caption"], caption)) > args.max_dropped_words:
            return None
        caption = _strip_exactly(job["caption"], caption)
        if "exactly" in caption.lower() and "exactly" not in job["caption"].lower():  # fact wording leaking through
            return None
        if _AFTERTHOUGHT.search(caption) and not _AFTERTHOUGHT.search(job["caption"]):  # "totaling nine flowers"
            return None
        # A paraphrase may add a word or two ("a sign reads ..." for a text fact); "A woman handles a black plastic
        # record near a box" adds a woman and a box.
        if len(_foreign_words(caption, job["caption"], job["selected"])) > 3:
            return None
        if any(quoted not in caption for quoted in _quoted(job["caption"])):  # quoted text is kept verbatim
            return None
        if any(phrase not in caption.lower() for phrase in _relation_phrases(job["caption"])):  # "lamp to the right of the sofa" stays
            return None
        # Numbers go in as words. A count written as a digit ("2 pillows") is spelled out rather than rejected:
        # the model repeats the digits on every retry and the row loses all its counts.
        caption = _spell_counts(job["caption"], caption, job["selected"])
        if _new_digits(job["caption"], caption, job["selected"]):  # "5 wheels" of no fact
            return None
        by_id = {fact["id"]: fact for fact in job["selected"]}
        used = [u for u in value.get("used") or [] if u in by_id]
        job["_write_tries"] = job.get("_write_tries", 0) + 1
        if not used:  # nothing verified went in, so nothing may change (retried a few times first)
            if job["_write_tries"] <= args.max_retries:
                return None
            return {"caption": job["caption"], "short_caption": job["short_caption"], "self_reflect": None, "correction": None, "used": []}
        short = None
        if job["short_caption"]:
            short = _clean_caption(value.get("short_caption"))
            # A short caption has few content words, so one paraphrased word costs a lot of coverage.
            if not short or coverage(job["short_caption"], [short]) < args.min_coverage - 0.1:
                return None
            if len(short.split()) > len(job["short_caption"].split()) + args.max_added_words // 2:
                return None
            if len(dropped_words(job["short_caption"], short)) > args.max_dropped_words + 1:
                return None
        elif args.synth_short:
            # A new prompt-style short caption: bounded, no digits, and its content words all come from the
            # caption or the facts (no invented objects).
            short = _clean_caption(value.get("short_caption"))
            if short:
                short = _strip_exactly(job["caption"], _spell_counts(job["caption"], short, job["selected"]))
            if not short or not 3 <= len(short.split()) <= args.short_max_words or re.search(r"\d", short):
                return None
            # "resting", "snow-covered" for "rests", "covering" are fine; "a red top" is not.
            if len(_foreign_words(short, job["caption"], job["selected"])) > 2:
                return None
        for fact_id in used:
            fact = by_id[fact_id]
            if fact["type"] in ("count", "part") and isinstance(fact["value"], int):
                count = fact["value"]
                if not re.search(rf"(?i)\b({_number(count)}|{count})\b", caption) and not (count == 1 and re.search(r"(?i)\b(a|an|one|single)\b", caption)):
                    return None
            if fact["type"] == "text" and fact["string"].lower() not in caption.lower():
                return None
        # Facts get lost in rewriting (SpatialRGPT retries until every region tag is used): retry a rewrite that
        # dropped most of them, and keep the last attempt whatever it used.
        if len(used) < args.min_used_fraction * len(by_id) and job["_write_tries"] <= args.max_retries:
            return None
        # The gpt turn: the Self-reflect and Correction carry the same facts. A critique that fails validation is
        # retried once with the rest; after that the caption is kept and the critique left as it was (None).
        critique = {}
        for key in ("self_reflect", "correction"):
            critique[key] = None
            if not job[key] or not args.enrich_critique:
                continue
            new = _enriched_critique(job[key], value.get(key), job, [by_id[u] for u in used], args)
            if new is None and job["_write_tries"] <= 1:  # one retry; the caption's own retries are not spent on it
                return None
            critique[key] = new
        return {"caption": caption, "short_caption": short, **critique, "used": used}
    return parse


_CRITIQUE_GLUE = [re.compile(r"(?i)\b" + phrase + r"\b") for phrase in
                  ("keep(?:ing|s)?", "while", "as described", "indeed", "remains?", "remaining", "still", "already", "intact", "unchanged", "as well", "also")]


def _enriched_critique(original: str, new, job: dict, used: list[dict], args) -> str | None:
    """The rewritten Self-reflect or Correction when it keeps the original and only adds the used facts to objects
    the original already names; None otherwise. Unchanged text is accepted (not every fact fits a critique)."""
    new = _clean_caption(new)
    if not new or coverage(original, [new]) < args.min_coverage:
        return None
    if len(new.split()) > len(original.split()) + args.max_added_words // 2:
        return None
    if dropped_words(original, new):  # "a grey beanie" -> "a gray hat" swaps the edit's target
        return None
    new = _strip_exactly(original, new)
    if "exactly" in new.lower() and "exactly" not in original.lower():
        return None
    # Only the facts' own words may come in: the critique names its objects already.
    if _foreign_words(new, original, used):
        return None
    # A fact about the property the critique edits ("silver instead of gold" + "the teapot is gold") contradicts
    # it: its value may not be repeated. Checked over every selected fact: the model copies facts it did not list.
    for fact in job["selected"]:
        value = fact.get("value") if fact["type"] != "count" else _number(fact["value"]) if isinstance(fact["value"], int) else None
        if isinstance(value, str) and value and len(re.findall(rf"(?i)\b{re.escape(value)}\b", new)) > len(re.findall(rf"(?i)\b{re.escape(value)}\b", original)) > 0:
            return None
    # No "keeping the two benches", "while the camel is indeed facing left as described" clauses.
    if any(len(glue.findall(new)) > len(glue.findall(original)) for glue in _CRITIQUE_GLUE):
        return None
    # No object the original critique does not name (a fact's noun, singular or plural).
    before, after = set(_words(original)), set(_words(new))
    for fact in used:
        for noun in (fact.get("objects") or []) + ([fact["item"]] if fact.get("item") else []):  # a worn item is an object too
            forms = {noun.lower(), _pluralize(noun).lower()} | {w.lower() for w in noun.split()}
            if forms & after and not forms & before:
                return None
    if any(quoted not in new for quoted in _quoted(original)):
        return None
    if any(phrase not in new.lower() for phrase in _relation_phrases(original)):
        return None
    new = _spell_counts(original, new, used)
    if _new_digits(original, new, used):
        return None
    return new


# ----------------------------------------------------------------------------------------------------- jobs

def conv_columns(parquet_file: pq.ParquetFile, prefix: str) -> list[tuple[str, str]]:
    """[(output column, source column)]: the main conversations first, then the optional short ones."""
    names = parquet_file.schema_arrow.names
    columns = []
    for name in (CONV_COLUMN,) + SHORT_COLUMNS:
        if prefix + name in names:
            columns.append((name, prefix + name))
        elif name in names:
            columns.append((name, name))
    if not columns or columns[0][0] != CONV_COLUMN:
        raise ValueError(f"{parquet_file} has no {CONV_COLUMN!r} column")
    return columns


def make_job(row: dict, columns, version: str, args) -> dict | None:
    turns = _as_list(row[columns[0][1]])
    caption = human_caption(turns)
    images = row.get(IMAGE_COLUMN) or []
    if caption is None or not images:
        return None
    short_turns = _as_list(row[columns[1][1]]) if len(columns) > 1 and row.get(columns[1][1]) is not None else None
    self_reflect, correction = critique_of(turns)
    return {
        "key": row_key(version, row.get(ID_COLUMN), turns, short_turns),
        "id": row.get(ID_COLUMN), "caption": caption,
        "short_caption": human_caption(short_turns) if short_turns else None,
        "self_reflect": self_reflect, "correction": correction,
        "images": images,
    }


def process_jobs(jobs: list[dict], vlm: VLM, perception: Perception, args) -> list[dict]:
    for job in jobs:
        job["final"] = decode_image(job["images"][-1])

    timings, started = {}, time.time()
    vocab_results = vlm.run(jobs, lambda job: build_vocab_messages(job, args), parse_vocab, args.vocab_max_new_tokens, "vocab")
    timings["vocab"], started = time.time() - started, time.time()
    records = {}
    for index, job in enumerate(jobs):
        vocab, raw, attempts = vocab_results[index]
        record = {"key": job["key"], "id": job["id"], "caption": job["caption"], "short_caption": job["short_caption"],
                  "vocab": vocab, "facts": [], "selected": [], "result": None}
        records[job["key"]] = record
        if vocab is None:
            record.update(status="failed", stage="vocab", raw=raw, attempts=attempts)
            continue
        job["vocab"] = vocab
    perception.prepare_chunk([job for job in jobs if "vocab" in job])
    for job in jobs:
        if "vocab" not in job:
            continue
        vocab = job["vocab"]
        job["scene"] = perception.analyze(job["final"], vocab)
        job["counts"] = {name: (e["count"], e["boxes_match"]) for name, e in job["scene"]["objects"].items()}
        # One generator per row, seeded by its id: crop order and the fact wording are reproducible.
        job["rng"] = random.Random(f"{args.seed}:{job['id']}:{job['key']}")
        job["regions"] = select_regions(job["scene"], vocab, job["counts"], args) if args.max_regions else []
        if job["regions"]:
            prepare_regions(job, job["rng"], args)
    timings["perception"], started = time.time() - started, time.time()

    live = [job for job in jobs if "scene" in job]
    describe_regions(live, vlm, args)
    timings["regions"], started = time.time() - started, time.time()

    to_write = []
    for job in live:
        record = records[job["key"]]
        scene, vocab, rng = job["scene"], job["vocab"], job["rng"]
        candidates = [dict(fact, key=key) for key, fact in
                      derive_facts(scene, vocab, job["counts"], args, rng, job["caption"]).items()]
        # A few extra so that a vetoed fact leaves the row with max_facts.
        selected = select_facts(candidates, rng, args, job["caption"])[:args.max_facts + (2 if args.veto else 0)]
        for number, fact in enumerate(selected, 1):
            fact["id"] = f"F{number}"
        job["selected"] = selected
        record.update(facts=candidates, measured=_summary(scene, vocab, job["counts"]), regions=scene["regions"],
                      region_raw=job.get("region_raw"))
        if args.viz_dir:
            save_viz(Path(args.viz_dir).expanduser(), job, scene)
    if args.veto:
        veto_facts(live, vlm, args)
    timings["veto"], started = time.time() - started, time.time()
    for job in live:
        record = records[job["key"]]
        job["selected"] = job["selected"][:args.max_facts]
        record.update(selected=[f["key"] for f in job["selected"]], vetoed=[f["key"] for f in job.get("vetoed", [])],
                      veto=job.get("veto_answers"))
        if not job["selected"]:
            record.update(status="no_facts")
            continue
        to_write.append((job, record))

    if to_write:
        write_jobs = [job for job, _ in to_write]
        results = vlm.run(write_jobs, lambda job: build_write_messages(job, args), make_write_parser(args),
                          args.max_new_tokens, "write")
        for index, (job, record) in enumerate(to_write):
            result, raw, attempts = results[index]
            record.update(status="ok" if result else "failed", stage="write", result=result, attempts=attempts,
                          raw=None if result else raw, fact_text={f["id"]: f["text"] for f in job["selected"]})
    timings["write"] = time.time() - started
    logger.info("%d rows: %s", len(jobs), ", ".join(f"{k} {v:.0f}s" for k, v in timings.items()))
    for job in jobs:
        for key in ("final", "images", "scene", "regions", "region_views", "rng", "vetoed", "veto_answers"):
            job.pop(key, None)
    return list(records.values())


def _summary(scene: dict, vocab: dict, counts: dict) -> dict:
    items = {item["name"]: item for item in vocab["objects"]}
    return {
        "objects": {name: {"n": len(e["gdino"]["instances"]), "gray": e["gdino"]["gray"],
                           "sam3": len(e["sam3"]["instances"]), "sam3_gray": e["sam3"]["gray"], "kind": items[name]["kind"],
                           "count": counts[name][0], "boxes_match": counts[name][1],
                           "agree": e.get("agree"), "n_votes": e.get("n_votes"),
                           "boxes": [i["box"] for i in e["instances"]]}
                    for name, e in scene["objects"].items()},
        "absent": scene["absent"],
    }


def save_viz(viz_dir: Path, job: dict, scene: dict) -> None:
    viz_dir.mkdir(parents=True, exist_ok=True)
    sheet = job["final"].copy()
    draw = ImageDraw.Draw(sheet)
    for name, entry in scene["objects"].items():
        for instance in entry["instances"]:
            draw.rectangle(instance["box"], outline="red", width=3)
            label = f"{name} {instance['score']:.2f} d={instance.get('depth')} {instance.get('color', ['?'])[0]}"
            draw.text((instance["box"][0] + 3, instance["box"][1] + 3), label, fill="yellow")
    sheet.save(viz_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', str(job['id']))}.jpg", quality=85)


# ------------------------------------------------------------------------------------------------- cache

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


def generate_dir(files, cache, cache_path: Path, vlm, perception, args, version) -> list[dict]:
    """Enrich every uncached row of this rank's row groups; returns the records made or found for them."""
    seen, done, started = [], 0, time.time()
    global_group = -1
    with open(cache_path, "a") as handle:
        for path in files:
            parquet_file = pq.ParquetFile(path)
            columns = conv_columns(parquet_file, args.v1_prefix)
            read = [c for _, c in columns] + [n for n in (IMAGE_COLUMN, ID_COLUMN) if n in parquet_file.schema_arrow.names]
            for group in range(parquet_file.metadata.num_row_groups):
                global_group += 1
                if global_group % args.world_size != args.rank:
                    continue
                if args.limit is not None and len(seen) >= args.limit:
                    return seen
                rows = parquet_file.read_row_group(group, columns=read).to_pylist()
                jobs = [job for job in (make_job(row, columns, version, args) for row in rows) if job is not None]
                if args.ids:
                    jobs = [job for job in jobs if job["id"] in args.ids]
                if args.limit is not None:
                    jobs = jobs[:args.limit - len(seen)]
                seen.extend(cache[job["key"]] for job in jobs if job["key"] in cache)
                pending = [job for job in jobs if job["key"] not in cache]
                for start in range(0, len(pending), args.chunk_size):
                    for record in process_jobs(pending[start:start + args.chunk_size], vlm, perception, args):
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        cache[record["key"]] = record
                        seen.append(record)
                        done += 1
                    handle.flush()
                    logger.info("rank %d: %s group %d, %d rows enriched, %.3f rows/s", args.rank, path.name, group,
                                done, done / max(time.time() - started, 1e-6))
                del rows
    return seen


# ------------------------------------------------------------------------------------------------- shard rewrite

def _replace_caption(conversations, caption: str | None, critique: tuple[str | None, str | None] | None = None):
    """The conversations with the human caption replaced and, when `critique` holds a rewritten Self-reflect or
    Correction, the edit turn rebuilt around them (markers, draft image and final image kept as they were)."""
    turns = [dict(turn) for turn in _as_list(conversations)]
    if caption:
        for turn in turns:
            if turn.get("from") == "human":
                turn["value"] = caption
                break
    if critique and any(critique):
        found = find_edit_turn(turns)
        if found is not None:
            index, parts = found
            self_reflect, correction = critique
            text = parts["pre"]
            if parts["sr_marker"] is not None:
                text += f"{parts['sr_marker']} {self_reflect or parts['self_reflect']}\n"
            text += f"{parts['corr_marker']} {correction or parts['correction']}{parts['post']}"
            turns[index]["value"] = text
    return turns


def rewrite_file(path: Path, output_path: Path, cache: dict, version: str, args) -> dict:
    parquet_file = pq.ParquetFile(path)
    columns = conv_columns(parquet_file, args.v1_prefix)
    schema_in = parquet_file.schema_arrow
    touched = {name for out, src in columns for name in (out, args.v1_prefix + out, src)}
    out_fields = []
    for out, src in columns:
        conv_type = schema_in.field(src).type
        out_fields += [pa.field(out, conv_type), pa.field(args.v1_prefix + out, conv_type)]
    # A synthesized short-caption column where the data has none. Rows without a synthesized caption get a copy of
    # the (enriched) conversations: the loader picks the short column half the time and must never see a null.
    synth = args.synth_short and len(columns) == 1
    if synth:
        out_fields.append(pa.field(SHORT_COLUMNS[0], schema_in.field(columns[0][1]).type))
        touched.add(SHORT_COLUMNS[0])
    out_fields += [field for field in schema_in if field.name not in touched]
    metadata = {k: v for k, v in (schema_in.metadata or {}).items() if k != b"pandas"}
    schema = pa.schema(out_fields, metadata=metadata or None)
    compression = parquet_file.metadata.row_group(0).column(0).compression.lower() if parquet_file.metadata.num_row_groups else "snappy"

    stats = {"ok": 0, "no_facts": 0, "failed": 0, "missing": 0, "skipped": 0}
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    writer = pq.ParquetWriter(temporary, schema, compression=compression)
    try:
        for group in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(group)
            sources = {src: table.column(src).to_pylist() for _, src in columns}
            ids = table.column(ID_COLUMN).to_pylist() if ID_COLUMN in table.column_names else [None] * table.num_rows
            new = {out: [] for out, _ in columns}
            for row_index in range(table.num_rows):
                main = _as_list(sources[columns[0][1]][row_index])
                short = sources[columns[1][1]][row_index] if len(columns) > 1 else None
                short = _as_list(short) if short is not None else None
                record = cache.get(row_key(version, ids[row_index], main, short)) if human_caption(main) else None
                if human_caption(main) is None:
                    stats["skipped"] += 1
                else:
                    stats[record["status"] if record else "missing"] += 1
                result = record["result"] if record and record["status"] == "ok" else None
                critique = (result.get("self_reflect"), result.get("correction")) if result else None  # older caches: no keys
                new[columns[0][0]].append(_replace_caption(main, result and result["caption"], critique))
                if len(columns) > 1:
                    new[columns[1][0]].append(None if short is None else _replace_caption(short, result and result["short_caption"], critique))
                elif synth:
                    new.setdefault(SHORT_COLUMNS[0], []).append(
                        _replace_caption(main, (result and result["short_caption"]) or (result and result["caption"]), critique))
            data = {name: table.column(name) for name in table.column_names if name not in touched}
            for out, src in columns:
                data[out] = pa.array(new[out], type=schema.field(out).type)
                data[args.v1_prefix + out] = table.column(src)
            if synth:
                data[SHORT_COLUMNS[0]] = pa.array(new[SHORT_COLUMNS[0]], type=schema.field(SHORT_COLUMNS[0]).type)
            writer.write_table(pa.table({f.name: data[f.name] for f in schema}, schema=schema), row_group_size=table.num_rows)
            del table, sources
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


def print_record(record: dict) -> None:
    print("=" * 100)
    print("ID          :", record["id"], "|", record["status"], "| vocab:", record.get("vocab"))
    print("CAPTION     :", record["caption"])
    if record.get("short_caption"):
        print("SHORT       :", record["short_caption"])
    measured = record.get("measured")
    if measured:
        counts = {n: f"{o['count']}{'' if o['boxes_match'] else '*'} (sam3 {o.get('sam3')}/gray {o.get('sam3_gray')}, gdino {o['n']}/gray {o['gray']})"
                  for n, o in measured["objects"].items()}
        print("MEASURED    :", counts, "absent:", measured["absent"])
    for region in record.get("regions") or []:
        print("REGION      :", {k: v for k, v in region.items() if v not in (None, [], {})})
    for fact in record.get("facts", []):
        mark = "*" if fact["key"] in record.get("selected", []) else "x" if fact["key"] in record.get("vetoed", []) else " "
        print(f"  {mark} [{fact['type']}] {fact['text']}")
    if record.get("result"):
        print("-" * 40, "used:", [record["fact_text"][u] for u in record["result"]["used"]])
        print("NEW CAPTION :", record["result"]["caption"])
        if record["result"].get("short_caption"):
            print("NEW SHORT   :", record["result"]["short_caption"])
        if record["result"].get("self_reflect"):
            print("NEW REFLECT :", record["result"]["self_reflect"])
        if record["result"].get("correction"):
            print("NEW CORRECT :", record["result"]["correction"])
    elif record.get("raw"):
        print("RAW REPLY   :", record["raw"])


def process_dir(input_dir: Path, vlm, perception, args, version: str) -> None:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {input_dir}")
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else input_dir / ".enrich_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # A marker left by an earlier run must not tell the other ranks this one is done before it has generated.
    (cache_dir / f"{version}_rank{args.rank}.done").unlink(missing_ok=True)
    cache = load_cache(cache_dir, version)
    records = generate_dir(files, cache, cache_dir / f"{version}_rank{args.rank}.jsonl", vlm, perception, args, version)
    status = {}
    for record in records:
        status[record["status"]] = status.get(record["status"], 0) + 1
    logger.info("rank %d, %s: %d rows, %s", args.rank, input_dir.name, len(records), status)

    if args.dry_run:
        for record in records:
            print_record(record)
        return
    if args.world_size > 1:
        wait_for_ranks(cache_dir, version, args)
        cache = load_cache(cache_dir, version)
    output_dir = Path(args.output_dir).expanduser() / input_dir.name if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_index, path in enumerate(files):
        if file_index % args.world_size != args.rank:
            continue
        stats = rewrite_file(path, output_dir / path.name, cache, version, args)
        logger.info("%s -> %s: %s", path.name, output_dir / path.name, stats)


def main(args) -> None:
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s rank{args.rank} %(levelname)s %(message)s")
    version = prompt_version(args)
    logger.info("prompt version %s", version)
    vlm = VLM(args)
    perception = Perception(args)
    for input_dir in args.input_dirs:
        process_dir(Path(input_dir).expanduser(), vlm, perception, args, version)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Enrich captions with detector/depth-grounded counts, relations, attributes.")
    parser.add_argument("--input_dirs", nargs="+", default=input_dirs)
    parser.add_argument("--output_dir", default=None,
                        help="Write shards to <output_dir>/<input dir name>/ instead of in place (default: in place).")
    parser.add_argument("--cache_dir", default=None, help="Default: <input dir>/.enrich_cache")
    parser.add_argument("--v1_prefix", default="v1-", help="Prefix of the column that keeps the source conversations.")
    parser.add_argument("--model_path", default=llm_model_path)
    parser.add_argument("--detector_path", default=detector_path)
    parser.add_argument("--depth_path", default=depth_path)
    parser.add_argument("--sam3_path", default=sam3_path, help="SAM 3: primary instance counter and mask source.")
    parser.add_argument("--sam3_low", type=float, default=0.3, help="SAM 3 instances below this are ignored.")
    parser.add_argument("--sam3_high", type=float, default=0.7,
                        help="SAM 3 instances at or above this count; any in [sam3_low, sam3_high) makes SAM 3 unsure.")
    parser.add_argument("--attribute_path", default=attribute_path)
    parser.add_argument("--device", default="auto", help="`device_map` for the VLM.")
    parser.add_argument("--attn_implementation", default=None, help="e.g. sdpa or flash_attention_2")
    parser.add_argument("--batch_size", type=int, default=8, help="VLM batch; halved automatically on CUDA OOM.")
    parser.add_argument("--chunk_size", type=int, default=64, help="Rows decoded and processed together (cache flush unit).")
    parser.add_argument("--det_batch_size", type=int, default=8, help="Grounding DINO batch; halved on CUDA OOM.")
    parser.add_argument("--vlm_max_pixels", type=int, default=640 * 640, help="Images are downscaled to this for the VLM.")
    parser.add_argument("--vocab_max_new_tokens", type=int, default=640, help="Pretty-printed JSON needs room.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # Qwen3.6 model card, instruct (non-thinking) mode.
    parser.add_argument("--temperature", type=float, default=0.7, help="0 for greedy decoding.")
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=1.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_retries", type=int, default=2)
    # Perception thresholds.
    parser.add_argument("--det_low", type=float, default=0.25, help="Grounding DINO boxes below this are ignored.")
    parser.add_argument("--det_high", type=float, default=0.35,
                        help="Boxes at or above this count; any box in [det_low, det_high) makes that count unreliable.")
    parser.add_argument("--attr_min_prob", type=float, default=0.4, help="SigLIP softmax prob (over 14 colors) for a color.")
    parser.add_argument("--attr_min_prob_material", type=float, default=0.7, help="... for a material.")
    parser.add_argument("--attr_min_prob_pattern", type=float, default=0.85, help="... for a pattern (the noisiest head).")
    parser.add_argument("--depth_margin", type=float, default=0.08, help="Normalised depth gap for in front of / behind.")
    parser.add_argument("--size_ratio", type=float, default=2.0, help="Box-area ratio for 'larger than'.")
    parser.add_argument("--max_count", type=int, default=10, help="Larger counts are not stated.")
    parser.add_argument("--min_box_frac", type=float, default=0.002,
                        help="Boxes smaller than this fraction of the image are not counted (and make the count unreliable).")
    # Region stage.
    parser.add_argument("--max_regions", type=int, default=6, help="Objects described from crops per image; 0 disables.")
    parser.add_argument("--region_scene_pixels", type=int, default=448 * 448, help="Scene image size in the region prompt.")
    parser.add_argument("--region_crop_pixels", type=int, default=336 * 336, help="Size of each crop in the region prompt.")
    parser.add_argument("--region_max_new_tokens", type=int, default=384)
    # Rewriting.
    parser.add_argument("--max_facts", type=int, default=6,
                        help="At most this many facts per row, taken in FACT_PRIORITY order.")
    parser.add_argument("--negation_prob", type=float, default=0.15, help="Share of rows offered a negation fact.")
    parser.add_argument("--max_position_facts", type=int, default=3, help="Position facts (2D, on, 3D) per row.")
    parser.add_argument("--max_3d_facts", type=int, default=1, help="In front of / behind facts per row (the noisiest).")
    parser.add_argument("--no_veto", dest="veto", action="store_false",
                        help="Skip the VLM check that drops selected facts it sees as false on the image.")
    parser.add_argument("--veto_counts", action="store_true", help="Let the VLM veto count facts too (VLMs miscount).")
    parser.add_argument("--veto_max_new_tokens", type=int, default=128)
    parser.add_argument("--no_critique", dest="enrich_critique", action="store_false",
                        help="Leave the Self-reflect and Correction of the gpt turn as they are (default: the used facts go in there too).")
    parser.add_argument("--synth_short", action="store_true",
                        help="Datasets without a short caption get a synthesized prompt-style one (conversations_short).")
    parser.add_argument("--short_max_words", type=int, default=30, help="Length bound of a synthesized short caption.")
    parser.add_argument("--max_dropped_words", type=int, default=1,
                        help="Reject rewrites that drop more than this many content words of the original.")
    parser.add_argument("--min_coverage", type=float, default=0.85,
                        help="Reject rewrites that keep less than this share of the original caption's content words.")
    parser.add_argument("--max_added_words", type=int, default=60)
    parser.add_argument("--min_used_fraction", type=float, default=0.5,
                        help="Retry a rewrite that uses fewer of the selected facts than this (the last try is kept).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N rows (of this rank) per input dir.")
    parser.add_argument("--dry_run", action="store_true", help="Print before/after and do not write shards.")
    parser.add_argument("--ids", nargs="+", default=None, help="Only these row ids (debugging; needs --dry_run).")
    parser.add_argument("--viz_dir", default=None, help="Save the final image with the kept boxes (debugging).")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    args = parser.parse_args(argv)
    if args.ids and not args.dry_run:
        parser.error("--ids only makes sense with --dry_run")
    if args.limit is not None and not args.dry_run and args.output_dir is None:
        parser.error("--limit without --dry_run would rewrite shards in place with most rows unprocessed; "
                     "add --dry_run or --output_dir")
    return args


if __name__ == "__main__":
    main(parse_args())
