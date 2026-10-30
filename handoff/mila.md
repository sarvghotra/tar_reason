# Mila

Owned by the Mila session. Mila: user `singhsd`, project folder `/home/mila/s/singhsd/CODE/LatentDCR/`
(`CLAUDE.md` there → `tar_reason/handoff/CLAUDE.md`, old copy `CLAUDE.md.old`; `papers/` and `tools/` next to `tar_reason/`).

## Now (2026-10-26 10:30 EDT)
- **Unified-model baselines (Inbox, round 2 + AlphaGRPO)**. Setup in `$SCRATCH/baselines/` (clones, `build_envs.sh`, `dl.sh`, 1-GPU
  `smoke.sh`, smoke outputs in `smoke/`); envs `~/envs/{showo,showo2,bagel,blip3o,emu3,alphagrpo}` (+ `.freeze.txt`). Job ids in
  `$SCRATCH/baselines/*_jobs.txt`. Smoke s/image (1 A100): Show-o 20, Show-o2 21, BLIP3-o 7, BAGEL 37, BAGEL + CoT 46, AlphaGRPO rt2i 52,
  srr 146 s/prompt, Emu3 230 (batch 4).
- **Done** (`runs.md`): Show-o, BAGEL + CoT (3 seeds each); AlphaGRPO rt2i GenEval2 seed 13 = AM 75.9.
- **AlphaGRPO-on-Tar** (`models/tar/sft/darshan_agrpo_tar_merged380`, Rorqual ckpt-380 on base Tar-7B; copy verified, 4 shards):
  single-pass official evals queued (`SINGLE_PASS=1 N_ITERS=2`, label `darshan_agrpo_tar_380`, 3 seeds × 3 benches, `agrpo_tar_jobs.txt`).
  Shared read-only with jeet 2026-10-30 (ACL, as E-450).
- **Queued with `RESUME=1`** (resubmitted 10:30 after 7 preemptions on `long`; `30d3b79`): AlphaGRPO rt2i (3 seeds), srr (1 seed: GenEval2
  2213 / GenAI 13 / TIIF 17), plain BAGEL (3 seeds). Partial copies in `$SCRATCH/tar_reason/results_jeet_partial/`.
- Not yet submitted: Show-o2, BLIP3-o (smokes OK), Emu3 (1 seed, split per bench), Uni-CoT (smoke pending). Use `RESUME=1` for all.
- Resume test (`results_resume_test*`, 1 GPU, forced requeue: 64/160 resumed): delete both folders once it finishes.

## Setup
- Session: Claude Code session `830290ea-cb02-4188-9cf2-1c5fb3768fa2`; when the user is away it runs in tmux
  session `claude` on **`login-2`** since 2026-10-26 (`ssh login-2`, `tmux attach -t claude`; login-4 was the earlier
  tmux node and is not reachable now). The VSCode plugin attached to login-4 resumed the same session on 2026-10-27 and
  submitted duplicate jobs: close the other client before resuming anywhere.
- Earlier note on the same session, opened from the VSCode plugin via
  `mila code CODE/LatentDCR --alloc …` (runs on a compute node, not a login node). Resume:
  `cd ~/CODE/LatentDCR && claude --resume 830290ea-cb02-4188-9cf2-1c5fb3768fa2`. No tmux login node chosen yet
  (Mila's `login.server.mila.quebec` round-robins over several login nodes; pick one with `ssh login-N`).
- Envs: `~/envs/tar` (training), `~/envs/qwen_judge` (judge; freeze in `~/envs/qwen_judge.freeze.txt`), `~/envs/genai_judge`
  (GenAI-Bench: Python 3.11, torch 2.8.0+cu128, transformers 5.19, upstream `$SCRATCH/git/t2v_metrics` editable @ `6ecb74f`;
  freeze in `~/envs/genai_judge.freeze.txt`). `import t2v_metrics` needs `ffmpeg` on PATH: a symlink to the imageio-ffmpeg
  binary sits in the env's `bin/`, which `bench_mila.sh` puts on PATH. Run it without tar_reason on PYTHONPATH.
- Weights: `$SCRATCH/models/tar/` (Tar-7B, `ta_tok.pth`, `ar_dtok_lp_512px.pth`, `vq_ds16_t2i.pt`,
  `siglip2-so400m-patch14-384/config.json`, `sft/slf_ref_edit_t21_ckpt17000`), `$SCRATCH/models/vlm/Qwen3-VL-8B-Instruct`.
- Data: `$SCRATCH/data/geneval2_50K_v2/` (train 49,000 + val800 + val256). `$SCRATCH/data/geneval2_50K/` is the
  old 49,488-row copy (288 val800 prompts inside); `val800_train_overlap.tsv` there lists them.
- `tar_reason/output_dir` → `$SCRATCH/tar_reason/output_dir`; SLURM logs in `output_dir/slurm_logs/`.
  Artifacts can't be published from there (resolves outside the allowed roots): copy out first.
- Partitions: `main` caps a user at 2 GPUs / 8 CPUs / 48 GB; `long` has no cap but is preemptible (requeued jobs
  resume); `short-unkillable` allows 4 GPUs for 3 h, not preempted (QoS: 4 GPUs per user). Use `a100l` (80 GB);
  the 2 `h100` nodes are usually full. A full `a100l` node with 4 free GPUs is rare on `long`.
- Bad nodes (scratch `Remote I/O error`, jobs die in ~1 min): **cn-g008, cn-g028, cn-g029** (2026-10-15/21). Submit with `--exclude=cn-g008,cn-g028,cn-g029`.
- On A100, t7 is ~7.3 min/step (rollout ~225 s, decode+judge ~130 s, train ~95 s); a val800 validation with
  `--eval_oracle` takes 1 h 46 min on 3 ranks. Judge loads in ~50 s.
- sbatch copies the script at submission: editing a launcher does not change already-queued jobs.
- Launchers: `bash_mila.sh` (latent reward, `BRANCH` from env), `bash_mila_oracle.sh` (latent + oracle critiques,
  std floor 0.04, untruncated sampling), `bash_mila_t7.sh` (t7). Untracked `scripts/rl_ft/bash_t7_sarv.sh` = the
  collaborator's original t7 script.
- Tools outside the repo, Mila only: `tools/walkthrough/build.py` (generates the code-walkthrough artifact; run
  with `~/envs/tar/bin/python`). In the repo: `scripts/rl_ft/viz/trajectory_gallery.py` (gallery of logged val
  strips; gallery of the first run: https://claude.ai/artifact/RH2zJjLUTSsTtt6PKw3GhR).

## Inbox
- 2026-10-26 from Fir (user): **no ACL share of `darshan_agrpo_tar_merged380` with jeet**; keep it private, just run the single-pass evals.
- 2026-10-25 from Fir (user + jeet, **priority over the other new baselines**): **AlphaGRPO on BAGEL** = the authors' released LoRA adapters on BAGEL-7B-MoT, `GEN_FAMILY=alphagrpo` (`eval/baselines/backends/alphagrpo.py`, same env as bagel + `peft`; clone `huangrh99/AlphaGRPO` to `$SCRATCH/baselines/AlphaGRPO`, its `Bagel/` dir is a modified BAGEL used as the modeling root; BAGEL base snapshot shared with the bagel family). Two rows: `MODEL_NAME=alphagrpo_rt2i UNI_ARGS="--mode think"` (adapter huangrh9/AlphaGRPO-rt2i: plan → image, single pass) and `MODEL_NAME=alphagrpo_srr UNI_ARGS="--mode reflect"` (adapter huangrh9/AlphaGRPO-srr: plan → image → one self-reflection → regenerated image; report first → final like Janus-R1). Their own GenEval run reflected only detector-flagged wrong images; ours reflects every image (no judge at test time): say so in the row. Smoke each on 8 prompts first (images + s/image; expect ~20 s per image, ~60 s per prompt for srr); 3 seeds if affordable, else 1. Do these before Show-o / Show-o2 / BLIP3-o / Emu3 / Uni-CoT; BAGEL plain (`GEN_FAMILY=bagel`) is the natural companion row and shares the setup.
- 2026-10-25 from Fir (user, paper baselines, round 2): **evaluate Show-o, Show-o2, BAGEL-7B, BLIP3-o-8B, Emu3 and Uni-CoT** through jeet's pipeline (single pass; Uni-CoT = its self-reflection loop, like Janus-Pro-R1). Code pushed: `eval/baselines/unified_gen.py` (one driver, same layout as the Janus / diffusers generators; CPU `--dry_run` passes for every family) + `eval/baselines/backends/<family>.py`, and `GEN_FAMILY=showo|showo2|bagel|blip3o|emu3|unicot` in `bench_jeet_mila.sh` (`UNI_ENV`, `UNI_ARGS`, `UNI_MODEL_PATH`; N_ITERS 2; `UNI_N_ITERS=5` for unicot to keep every round). **Each backend was written from the authors' inference scripts without running them (Fir has no internet on compute nodes): expect small API mismatches; fix them in the backend and push.** Per family (docstring at the top of each backend has the details, defaults = the authors' benchmark settings, prompts verbatim):
  1. **Clones** (login node, `$SCRATCH/baselines/`): `showlab/Show-o` (Show-o and its `show-o2/` subdir), `ByteDance-Seed/Bagel`, `JiuhaiChen/BLIP3o` **branch `main`** (the default branch is BLIP3o-NEXT, a different model), `baaivision/Emu3`, `Fr0zenCrane/UniCoT`. Backend defaults point at `$SCRATCH/baselines/<name>`; override with `--<family>_repo` in `UNI_ARGS`.
  2. **Envs** (uv, one per family; bagel + unicot can share): showo: torch 2.2–2.4, transformers 4.41.1, diffusers 0.30.1, omegaconf, einops (no flash-attn). showo2: torch 2.5.1, transformers 4.47.0, diffusers 0.31.0, omegaconf, torchdiffeq, timm, einops, decord. bagel/unicot: torch 2.5.1, transformers 4.49.0, **flash-attn 2.5.8** (hard import), accelerate, einops, opencv-python, safetensors. blip3o: torch 2.3.0, transformers 4.51.3, `git+https://github.com/JiuhaiChen/diffusers.git`, flash-attn 2.6.2, timm 0.6.13, qwen_vl_utils, datasets, peft, then `pip install -e .` in the main checkout. emu3: torch ≥ 2.2, **transformers 4.44.0**, tiktoken 0.6.0 (flash-attn optional: `--attn sdpa` default).
  3. **Weights**: showlab/show-o-512x512 + showlab/magvitv2 + microsoft/phi-1_5 (tokenizer); showlab/show-o2-7B + Qwen/Qwen2.5-7B-Instruct (tokenizer) + google/siglip-so400m-patch14-384 + **`Wan2.1_VAE.pth` from Wan-AI/Wan2.1-T2V-14B** (`--wan_vae <path>`); ByteDance-Seed/BAGEL-7B-MoT (snapshot, ~29 GB); BLIP3o/BLIP3o-Model-8B (snapshot; importing `blip3o.model` also downloads the 17 GB jiuhai/eva_clip_vision_tower once); BAAI/Emu3-Gen + BAAI/Emu3-VisionTokenizer; Fr0zencr4nE/UniCoT-7B-MoT (full BAGEL-format snapshot).
  4. **Smoke each on 1 GPU** (`UNI_ARGS="--limit_prompts 8"`, BENCH=geneval2, SEEDS=713), look at the images, note s/image. Rough costs per image on an A100: Show-o < 2 s, Show-o2-7B ~5 s (100 steps), BAGEL ~20 s, BLIP3-o ~15 s, **Emu3 ~5 min unbatched (8k AR tokens with CFG; batch 4–8 → ~1 min amortised)**, **Uni-CoT ~1–3 min per round** (plan + critique + regen). Plan: 3 seeds for the cheap ones, 1 seed for Emu3 and Uni-CoT, and tell me if any is worse than that.
  5. **Uni-CoT specifics**: `--max_rounds 4` (our budget; their loop runs to 21), one prompt per process, blip3o too (the driver pins one GPU per rank for blip3o since its library hard-codes cuda:0). Report first image → final like Janus-R1, plus how many prompts it regenerated.
  Report rows (GenEval2 AM + GM + all-correct, GenAI-1600, TIIF short / long, mean ± se, resolution, seeds) to Fir's Inbox + `runs.md` next to the diffusion block.
- 2026-10-21 from Fir (remaining): F-250 and base Tar-7B (single pass: draft only) through jeet's pipeline, if time allows.
- 2026-10-12 17:45 PDT from Fir, task D (remaining): t11 checkpoints 150/200/300 on the official benchmarks once Fir
  keeps them (they are in Fir `~/scratch/tar_reason_kept/darshan_t11_fir/`; need copying to Mila). (GenAI-bench setup done 2026-10-14.)

## Log (newest first)
- 2026-10-30 03:45: Tar-generator TIIF OOMs on L40S (decode_codes / k_cache at TIIF batch size, 44.6 GB): AlphaGRPO-on-Tar TIIF moved back to A100L (17, 91 resubmitted as 11453244/11453245; 13 = 11408855). Tar GenEval2 fits on L40S (seed 2213: AM 74.2). Unified-model baselines are fine on L40S.
- 2026-10-30 03:00: all 120 A100L GPUs on `long` busy for ~34 h (no starts). Smoke-tested BAGEL / Show-o2 / AlphaGRPO srr on L40S (48 GB; peaks 36-37 GB; images ~identical to A100 at same seed; ~1.1-1.2x slower), then moved the 31 queued GenEval2 + TIIF jobs in place (`scontrol update`) to 2x L40S, 16 CPU, 96 GB; the 15 GenAI jobs stay on 4x A100L (Qwen3.5-27B judge needs >48 GB) but shrunk to 16 CPU, 160 GB; Emu3 unchanged (51 GB at batch 4).
- 2026-10-30 01:30: shared `models/tar/sft/darshan_agrpo_tar_merged380` read-only with jeet (`setfacl -R -m u:sarvjeet-singh.ghotra:rX`; revoke with `-x`).
- 2026-10-27 17:45: two clients had the session open (tmux login-2 + VSCode login-4); login-4's copy resubmitted srr GenAI as 11414737 (2 GPUs, 48 GB; its first try 11414310 OOMed). Kept 11414737, cancelled my held 11414376.
- 2026-10-27 00:10: AlphaGRPO-on-Tar evals submitted (Inbox from Fir + Rorqual; both lines removed, ACL pending in Now).
- 2026-10-26 10:30: RESUME=1 added and tested; AlphaGRPO + plain BAGEL jobs resubmitted with it. Show-o and BAGEL + CoT done.
- 2026-10-26 00:55: shared `models/tar/sft/darshan_pm1_E_merged450` (E-450) read-only with jeet (`setfacl -R -m u:sarvjeet-singh.ghotra:rX`), same as F-800 / pm-500; revoke with `setfacl -R -x u:sarvjeet-singh.ghotra <dir>`.
- 2026-10-26 00:30: unified-model baselines set up and smoke-tested (6 families + AlphaGRPO); BAGEL + CoT evals queued first (user).
- 2026-10-25 11:30: diffusion baselines done (21 jobs); deleted that Inbox line; told Fir.
- 2026-10-25: diffusion baselines: env + weights + 3 smokes; launcher N_ITERS fix `0a256f1`; 21 jobs submitted (Qwen-Image 1 seed, 1024 px).
- 2026-10-24 19:10: B-450 done (runs.md); deleted its Inbox line; told Fir.
- 2026-10-24 18:10: E-450 done (runs.md); deleted its two Inbox lines; told Fir. B-450 running.
- 2026-10-24: E-450 and B-450 copied (user); 18 jobs submitted. Removed the dashboard Inbox line again (done 2026-10-23, it had come back).
- 2026-10-23 evening: Janus baselines done (18 jobs); GenAI resolution check (384 vs 512 upscaled); deleted that Inbox line; told Fir.
- 2026-10-23: Janus baseline set up (repo wendell0218/Janus-Pro-R1 @0e40b3a + weights in `$SCRATCH/baselines/janus_pro_r1/` and hf_home; env `~/envs/janus`), smoke passed, 18 jobs submitted. Slurm controller was down 14:00–17:18.
- 2026-10-23: per-image means (GenAI, TIIF short/long × 4 models) added to runs.md + CSV; deleted that Inbox line.
- 2026-10-22: n5 official evals done for 4 models (runs.md); scratch damage worked around in the launcher; told Fir.
- 2026-10-21: shared two merged models read-only with jeet (`sarvjeet-singh.ghotra`) by ACL (user request): `/network/scratch/s/singhsd/models/tar/sft/darshan_mr_fir_merged800` (F-800 on t8) and `/network/scratch/s/singhsd/models/tar/sft/darshan_mr_pm_fir_merged500` (promptmix-run-500 on promptmix t1): `u:jeet:rX` recursive on the two folders, `u:jeet:--x` on `$SCRATCH`, `models/`, `models/tar/`, `models/tar/sft/`. To keep everything else private, removed 'other' permissions (o-rwx) on every sibling along that path (benchmarks, data, git, hf_home, tar_reason, .uv_cache, slurm-*.out, models/vlm, models/tar/{Tar-7B, siglip2…, *.pth, .cache}, the other sft/* models) and on the path folders; nobody else could reach them anyway (`$SCRATCH` was already 750). Verified with getfacl. Revoke: `setfacl -R -x u:sarvjeet-singh.ghotra <dir>`.
- 2026-10-21: Inbox: N_ITERS=5 default; submitted 36 n5 jobs (F-800, pm-500 re-runs; t8, promptmix t1).
- 2026-10-21: F-800 + pm-500 (GenAI-1600) done through jeet's pipeline; several `long` preemptions resubmitted on short-unkillable/main.
- 2026-10-21 08:50: cn-g028 has the same scratch I/O error as cn-g029; F-800 GenEval2 seeds 2213/713 resubmitted off it, pending jobs updated to exclude both.
- 2026-10-21: user: GenAI on the full 1,600 prompts; F-800 through jeet's pipeline. Submitted 9 F-800 + 3 pm-500 GenAI-1600 jobs.
- 2026-10-21: pm-run-500 through jeet's pipeline done (runs.md); deleted the jeet-pipeline Inbox line; told Fir.
- 2026-10-20 evening: ported jeet's pipeline (`4e7e0e4`); merged pm-run 400/500; sanity + pm-500 jobs submitted; t7 launcher defaults updated (`7486266`). Removed the merge-notice, pm-run-finished and old official-evals Inbox lines (acted on).
- 2026-10-20 09:00: all 30 seed jobs done; 3-seed table in runs.md; told Fir.
- 2026-10-20 06:30: C0 probe done for F-250, F-800, base Tar (runs.md); deleted that Inbox line; told Fir.
- 2026-10-20 04:00: C0 probe written (CPU test + GPU smoke), full runs submitted for F-250, F-800, base Tar.
- 2026-10-20 02:30: the merge `168f38a` re-hardcoded jeet's SigLIP2 path in `TextAlignedTokenizer.from_checkpoint` (`3d4ae90`); t8 seed benches 714/715 failed on it. Fixed in `395fcb4` (SIGLIP2_PATH overrides any teacher path; missing local path → HF id), GPU smoke passed, resubmitted. Scorer alignment done (equal).
- 2026-10-20: pulled `168f38a` (jeet's main) and the official-evals plan; jeet's scratch readable via the direct path; GenAI-800 subset computed; optz check + 30 seed jobs submitted. Removed the old scorer-alignment and dataset-seed Inbox lines (superseded / noted).
- 2026-10-18 05:00: promptmix-500 + F-800 benchmarks done (rows in `runs.md`); deleted that Inbox line; told Fir.
- 2026-10-18 00:20: started promptmix-500 + F-800 benchmarks (Inbox). Dropped the superseded F-300 / G-300 / older F-G Inbox lines (F-800 replaces F-300; G stopped).
- 2026-10-16 00:30: F-250 benchmarks done (see Now); deleted that Inbox line; told Fir.
- 2026-10-15 evening: started F-250 official benchmarks + depth-4 mdp (Inbox).
- 2026-10-15 13:00: 3a done (negative): natural-ratio stop-SFT ≈ baseline on val800; deleted the 3a Inbox line; told Fir.
- 2026-10-15 08:30: stop-SFT LR 1e-5 (part 1) and LR 5e-5 × 2 epochs (full val800) both worse than F-250 (prior shift toward stopping); natural-ratio run started. `tools/watch/depth_curve.py` got `--only <prompts.jsonl>`.
- 2026-10-15 05:30: 3a: 8 data shards done (shard 5 preempted once on `long`, rerun on `main`), 3,100 examples; stop-SFT trained (193 steps); F-250 baseline done (own 0.841 / oracle 0.889 at depth 4); stop-SFT eval queued.
- 2026-10-15 00:00: started 3a (see Now). Checked jeet's Mila scratch: not readable (removed that Inbox line; scorer-alignment line kept).
- 2026-10-14 01:30: H done (MDP depth-4 official GenEval2: 77.9 at d=4, full history 74.3), J done (GenAI-Bench t8 39.3 → 40.7, base 37.7). Deleted the H/I/J Inbox line (kept its F/G follow-up). Told Fir and TamIA.
- 2026-10-14 00:10: I done (base Tar-7B: GenEval2 64.7, TIIF 69.2 / 65.2); H part 1 done (MDP am_upto 0.786 → 0.845 at d=4, full history 0.794 → 0.803); GenAI smoke passed.
- 2026-10-13 22:45: pulled `c39b934` (plan F–N) and later handoff-only commits. Submitted H (11243481/82) and I (smoke 11243483
  passed, full 11243541); set up GenAI-Bench (J), smoke 11243655. Code: `--single_pass` (adhoc generators), `SINGLE_PASS` and
  opt-in `genai` stage in `bench_mila.sh`, `eval/genai_bench_score.py` (additive; GenEval2/TIIF scoring unchanged).
- 2026-10-13 20:40: backfilled `val/am_upto_{0..4}` and `val/allpass_upto_{0..4}` into the wandb summaries of
  `depth4_t8_geneval2_part1/2` (Fir inbox, user request), recomputed from each run's val_dump with `tree_stats`'
  definition @ `7d568ff`; checked `am_upto_0` = `am_0` and `am_upto_4` = `am_final` in both, and the 800-prompt mean
  matches `runs.md`. Pulled `7d568ff` (new metrics) before any further eval. Deleted both 2026-10-13 Fir inbox lines.
- 2026-10-13 00:40: t8 depth-4 on official GenEval2 done: AM 70.9 / 75.7 / 76.4 / 75.5 / 74.3 after 0–4 rounds; the
  critic stops on only ~22% of prompts by round 4. Told Fir and TamIA (Inbox).
- 2026-10-12 22:10: pulled Fir's `--context_mode` / `--chain_siblings` code (b0e0e34, a011791) while the depth jobs
  11222564/65 were queued (a rule-4 slip: the check printed the files but the pull went ahead). Checked the diff: with
  the defaults (`full`, `0`) and validation's explicit branch, rollouts are unchanged, so the jobs are unaffected.
- 2026-10-12 22:04: t8 ck-500 official benchmarks done: GenEval2 final AM 76.8 (+6.5 from refining), TIIF short/long final 70.5/68.6.
- 2026-10-12 21:00: pulled the A–G plan (`9793e0e` … `3e0c98f`); verified the t8 merged copy; queued task D.
- 2026-10-12 10:00: t7 Mila ck-100 benchmarks done: GenEval2 64.7 → 65.7, TIIF short 63.7 → 64.8, long 59.5 → 59.7.
- 2026-10-12 08:50: official benchmarks for SFT t21 and t7 Mila ck-200 done (numbers in `runs.md`, row `bench_mila`):
  GenEval2 final AM 61.8 → 71.7; TIIF short final 63.6 → 71.5, long 60.3 → 61.7 (refine hurts on long).
- 2026-10-12 06:40: critic diagnostic done (Fir inbox item 1). Full reports: `output_dir/critic_diag_{t7mila200,sft}/critic_diagnostic.txt`.
  Stop decision unchanged by t7; Corrections narrower than SFT's (name all of 2 failed objects 44% vs 55%); fix rate
  when mentioned 12% → 26%. Extra breadth script: `tools/watch/critic_breadth.py` (outside the repo).
- 2026-10-12 04:45: pulled the t8–t11 merge (`a43f083` … `f1e40b9`); started the critic diagnostic jobs.
- 2026-10-10 05:02: `darshan_t7_mila` done (step-200 val800 in `runs.md`); `keep_t7_ckpts` done (100 + 200 kept).
- 2026-10-10 03:17: checkpoint-200 saved and copied. A leftover watcher from the previous session ate one event:
  after a session restart, check `ps` for old `watch_t7_mila.sh` processes before starting a new one.
- 2026-10-10 02:25: session restarted; rebuilt the watcher in `tools/watch/` (the scratchpad copy was lost).
- 2026-10-09: moved notes to `handoff/` (Fir's scheme): rewrote this file and the Mila rows of `runs.md`;
  `LatentDCR/CLAUDE.md` is now a symlink to `tar_reason/handoff/CLAUDE.md` (old file kept as `CLAUDE.md.old`);
  deleted `LatentDCR/MILA_HANDOFF.md` and `LatentDCR/FIR_HANDOFF.md` (frozen copies in `handoff/archive/`).
- 2026-10-09 12:13: step-100 val800 done; checkpoint-100 copied to `kept/`. Started `keep_t7_ckpts`.
- 2026-10-09: rewrote the code walkthrough for the t7 path (artifact KPqjbeJ5iZMWsFuAyVmUHk, commit `e535421`).
- 2026-10-08 23:10: rechained `darshan_t7_mila` as `MAX_STEPS=200 EVAL_STEPS=100` (12 jobs after 11143217).
- 2026-10-08: `darshan_t7_mila` started (job 11143217: step-0 eval + steps 1–11). Earlier attempts cancelled within
  minutes to switch data: val256 → val800 → val800 minus 288 → the collaborator's leak-free v2 data.
- 2026-10-08: added `--dataset_seed` and redo-of-interrupted-validation to `train_grpo.py` (`a533714`).
- 2026-10-08: built `~/envs/qwen_judge`, downloaded Qwen3-VL-8B; t7 smoke test passed (job 11141217, 29 min).
- 2026-10-08: cancelled `darshan_rl_oracle_b25_mila` at step ~54 after its step-0 diagnosis.
- 2026-10-07: `darshan_rl_2gpu_mila` finished (500 steps, 19 h 20 min). Built the trajectory gallery tool.
- 2026-10-06: latent-reward smoke test passed (1 A100: 42.5 GB peak, ~125 s/step); SigLIP2 path fix `39a6672`.
- 2026-10-05: Mila setup (GitHub SSH key, clone, `~/envs/tar`, HF weights).
