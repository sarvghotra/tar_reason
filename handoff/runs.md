# RL runs (all clusters)

One row per run. Each session edits only the rows for its own cluster. Longer notes go in that cluster's
`<cluster>.md` Log. Status: `planned`, `queued`, `running`, `done`, `failed`, `cancelled`.

| Run (`RUN_NAME`) | Cluster | Recipe | Batch | Status | Key results | Detail |
|---|---|---|---|---|---|---|
| `darshan_rl_2gpu_mila` | Mila | latent reward (frozen Tar-7B), own critiques (trained), `BRANCH=4,2`, LR 1e-5 cosine, KL 0.01, 1 PPO epoch, `stop_penalty=0`, from t21 @17K, old data + val256 | 4/step (2× A100 × 2) | done 2026-10-07, 500 steps, job 11105481 | val256: draft AM 0.726 → 0.742 (peak 0.750 @400), final AM 0.720 → 0.745; final − draft −0.012…+0.003; fix ~12% throughout; KL ≤ 0.008. Drafting improved, refining didn't. | ckpts 450/475/500 in Mila `output_dir/darshan_rl_2gpu_mila/`; gallery https://claude.ai/artifact/RH2zJjLUTSsTtt6PKw3GhR |
| `darshan_rl_oracle_b25_mila` | Mila | latent reward, oracle critiques (pasted), `BRANCH=2,5`, `adv_std_floor 0.04`, untruncated image sampling, old data + val256 | 4/step (2× A100 × 2) | cancelled at step ~54, 2026-10-08 | step 0 val256: fix 16.3% own critique vs 17.1% oracle (by skill, oracle helps only attribute 20→38%; count 13→11%) → refiner is the bottleneck, not the critic | `mila.md` |
| `darshan_t7_mila` | Mila | t7: Qwen3-VL-8B pixel judge (GPU 3), hybrid critiques, `BRANCH=2,4`, LR 1e-4 const, KL 0.05, 2 PPO epochs, `draft_reward children`, stop 0.5@0.8, `DATASET_SEED=19`, data v2 + val800 | 18/step (3 A100 × 6) | done 2026-10-10 05:02, 200 steps (`MAX_STEPS=200`, evals at 0/100/200), jobs 11143217 + 11145380…91 | **val800, own critique (step 0 → 100 → 200):** draft AM 0.676 → 0.704 → **0.746**; final AM 0.666 → 0.718 → **0.764**; final − draft −0.010 → +0.013 → **+0.018**; fix 12.0% → 21.2% → **25.5%**; brk 8.9% → 7.5% → 7.1%; improved/degraded 16.5/21.9% → 22.5/16.6%; looks_good 12.2% → 15.8%. **Oracle critique:** final − draft +0.009 → +0.035 → **+0.056**; fix 14.9% → 28.5% → **36.5%**; brk 7.4% → 8.8% → 6.4%. Step-200 fix by skill, own/oracle: object 47/67%, count 22.5/26%, attribute 22.5/55%, position 26/33%, verb 15/20% (step 0: 16/18, 12/15, 8.5/14, 13/14, 7/5). One-step KL spikes at 13–14, 55 (3.5, grad 87), 186 (0.61), all recovered; KL baseline ~0.1–0.2 late. ~7.3 min/step. | `mila.md`; checkpoint-100 and -200 in Mila `$SCRATCH/tar_reason/kept/darshan_t7_mila/` |
| `critic_diag_t7mila200`, `critic_diag_sft` | Mila | eval only (`MAX_STEPS=0 EVAL_ORACLE=0`, `--val_dump_dir`), val800, own greedy critic, for t7 Mila checkpoint-200 and for SFT t21 | 3 A100 | done 2026-10-12 (jobs 11209613, 11209614; ~58 min each) | **SFT → t7 ck-200:** stop table (pass+stop / pass+refine / fail+stop / fail+refine) 56/59/42/643 → 75/82/51/592; error recall 93.9% → 92.1%; stop precision 57.1% → 59.5%; stop recall 48.7% → 47.8%. Mention rate (keyword) 90.2% → 88.0%; fix rate mentioned 12.0% → **26.3%**, not mentioned 12.3% → 19.4%. **Breadth** (distinct failed objects named in the Correction, `tools/watch/critic_breadth.py` on Mila): 2 failed objects → both named 55.4% → **44.2%**; 3 → all named 32.6% → **22.5%**; single-clause Corrections 57% → 64%. The ck-200 eval reproduces the run's own step-200 val exactly (am 0.746 → 0.764, fix 25.5%). | `mila.md`; reports in Mila `output_dir/critic_diag_*/critic_diagnostic.txt` |
| `bench_mila` (official benchmarks) | Mila | `scripts/eval/bench_mila.sh`: adhoc generators (draft → one greedy critique → one refinement), seed 713; GenEval2 official 800 prompts + official `evaluation.py` (Qwen3-VL-8B, soft-TIFA); TIIF-Bench testmini (Qwen2.5-VL-7B judge, official summaries) | 4 A100, ~66 min/model | done 2026-10-12 for SFT t21, t7 Mila ck-100, ck-200 (jobs 11209951/53/55) | **GenEval2 AM draft → final:** SFT 62.0 → 61.8; ck-100 64.7 → 65.7; **ck-200 68.9 → 71.7**. GM final 18.4 / 19.0 / 24.9; all-correct final 12.8% / 11.6% / 16.0%. **TIIF overall (9 sub-attributes), draft → final, short | long:** SFT 61.9 → 63.6 | 59.6 → 60.3; ck-100 63.7 → 64.8 | 59.5 → 59.7; ck-200 68.0 → **71.5** | 63.2 → 61.7 (refining hurts on long). One seed each. | `$SCRATCH/tar_reason/bench/<model>/seed_713/summary.json`, TIIF reports in `tiif/eval_results/` |
| `t7_smoke_fir` | Fir | t7, 2 steps, eval on 24 prompts | 18/step (3 H100 × 6) | done 2026-10-09, job 63829085 | pass: judge 260 s, steps 0–2, ckpt 1.9 GB. Rollout ~170 s, reward ~81 s, train ~35 s = **~4.8 min/step** | `fir.md` |
| `darshan_t7_fir` | Fir | t7, same flags as `darshan_t7_mila`, 300 steps, val800 every 50 | 18/step (3 H100 × 6) | done 2026-10-11 00:22 PDT, 300 steps (jobs 63858143 SIGBUS @41 → 63858144 timeout @~250 → 63858145 `Done.`) | **val800, own critique** (step 0 / 50 / 100 / 150 / 200 / 250 / **300**): draft AM 0.670 / 0.695 / 0.717 / **0.742** / 0.684 / 0.689 / **0.698**; final AM 0.672 / 0.710 / 0.738 / 0.756 / 0.730 / 0.738 / **0.762**; final − draft +0.002 / +0.015 / +0.021 / +0.014 / +0.045 / +0.049 / **+0.064**; fix 13.9 / 24.5 / 24.5 / 24.4 / 30.5 / 34.2 / **38.0%**; brk 7.7 / 9.9 / 7.2 / 7.3 / 8.9 / 9.7 / **7.7%**; improved/degraded at 300 36.1/12.4%. **Oracle:** fix 14.5 → 25.1 → 28.9 → 38.2 → 41.2 → 43.5 → **51.2%**, final AM **0.798**. **Caveat:** after step 150 the draft got worse (0.742 → 0.684 at 200, 0.698 at 300) while final AM only returned to its step-150 level (0.756 → 0.762): most of the larger final − draft gap is a weaker draft, not a better final. KL median 0.007 (1–20) → 0.24 (141–160) → 0.4–0.7 (181–300), spikes up to 3.4 (grad 60) in 281–300, each recovered. ~5.1 min/step. | `fir.md`; checkpoint-100/200/300 kept in Fir `~/scratch/tar_reason_kept/darshan_t7_fir/` |
| `t7_smoke_tamia` | TamIA | t7, 2 steps, eval on 24 prompts | 18/step (3 H100 × 6) | done 2026-10-09, job 515432 | pass: judge 40 s, steps 0–2, ckpt 1.9 GB, 21 min total. Rollout ~155 s, reward ~75 s, train 33–63 s = **~4.5–5 min/step** | `tamia.md` |
| `darshan_t7_tamia` | TamIA | t7, same flags as `darshan_t7_fir`, 300 steps, val800 every 50 | 36/step (2 nodes × 3 H100 × 6) | **stopped at step 250** (516259 TIMEOUT @120 → 516260 TIMEOUT during eval 250 → 516261 cancelled by the user 2026-10-12); no step-250/300 eval | **val800, own critique** (step 0 / 50 / 100 / 150 / **200**): draft AM 0.675 / 0.696 / 0.727 / **0.744** / 0.663; final AM 0.682 / 0.697 / 0.742 / **0.763** / 0.735; final − draft +0.008 / +0.001 / +0.016 / +0.019 / **+0.072**; fix 14.9 / 18.7 / 22.0 / 21.7 / **35.5%**; brk 7.0 / 9.4 / 6.9 / 5.9 / 7.9%. **Oracle:** final − draft +0.011 / +0.017 / +0.056 / +0.072 / +0.117; fix 14.9 / 19.3 / 31.0 / 34.6 / **44.6%**; brk 7.7 / 8.3 / 7.2 / 5.4 / 8.4%. **Same caveat as Fir:** after step 150 the draft got worse (0.744 → 0.663) and final AM fell (0.763 → 0.735); the bigger final − draft gap is mostly a weaker draft. Train fix_1 (25-step means) 15% → 25% (76–100) → 33% (126–150) → 39–40% (201–250); reward_1 +0.01 → +0.06 → +0.11. KL: wobble 67–71, spike 88 (1.63, grad 39), baseline ~1–1.7 over 170–200, huge spike 217 (KL 59, grad 2993, clipped), ~0.6–0.8 at 226–250. ~4.9 min/step; val800 ~40 min. | `tamia.md`; ckpt-85/100 kept in TamIA `$SCRATCH/tar_reason_kept/darshan_t7_tamia/`; 150/200 pruned |
| `t11_smoke_tamia` | TamIA | t11 (critic-only, image tokens KL-anchored to merged t8), 2 steps, eval on 24 prompts | 9/step (1 node × 3 ranks × 3) | done 2026-10-12, job 612737 (19 min) | pass: judge 30 s. **Step-0 val (24 prompts):** own fix 0.412, final − draft +0.073, am_0 0.751 → am_final 0.824, brk 0.029; oracle fix 0.465, am_final 0.829 (≈ t8 step 500: 0.42 / 0.825 / 0.47). Steps 1–2: pg_txt −0.016 / +0.023, kl_img 0 / 0.0011, kl_txt 0 / 0.0008, grad 0.09 / 0.07. Rollout 66–99 s, reward 30–59 s, train 53–109 s ≈ 3.4–3.5 min/step. val_dump jsonl written. | `tamia.md` |
| `darshan_t11_tamia` | TamIA | t11, same flags as `darshan_t11_fir` (`DATASET_SEED=421`) | 36/step (4 nodes × 3 H100 × 3) | queued 2026-10-12 05:00 EDT (612742 → 612743 → 612744, 12 h each; smoke passed, 612742 pending on Priority) | – | `tamia.md` |
| `darshan_t11_fir` | Fir | t11 (collaborator): base + KL ref = t8 ckpt-500 merged into t21; fresh LoRA; critique tokens trained (`--critique_source policy`, K=3 critiques x M=4 images, critique advantage = mean of its images, `REFLECT_TOKEN_WEIGHT=20`); image tokens KL-only (`--image_adv_scale 0`); LR 1e-4 const, KL 0.05, 1 epoch, `BREAK_WEIGHT=2`, stop penalty 0, std floor 0.04, `DATASET_SEED=421` | 36/step (4 nodes x 3 H100 x 3) | stalled at step 123 since 2026-10-12 11:45 (SIGBUS in a judge at step 13, then in a trainer rank at step 123; spare 64518140 queued, will resume from ckpt-120) | **step 0 val800** (= merged t8): own fix 42.6%, brk 5.3%, delta_am +0.054, draft AM 0.773, final AM 0.828, looks_good on passing/failing drafts 0.45/0.09; oracle fix 45.2%, brk 6.5%, final 0.841 (jeet's t8 @500: 0.42 / 0.06 / +0.048 / 0.776 / 0.825 / 0.46/0.08; oracle 0.47 / 0.845): the merged base reproduces t8. **Step 50:** own fix 44.8%, brk 6.1%, delta_am +0.057, draft 0.770, final 0.827, looks_good pass/fail 0.55/0.11; oracle fix 48.6%, brk 6.3%, final 0.842. Anchor holds (draft and oracle unchanged); critic changes small (fix +2.2 pt, stops more on passing drafts but also slightly more on failing ones), final AM flat. Train kl_img 0.001 → 0.0066 (step 40) → 0.005 (50), kl_txt 0.06. **Step 100:** own fix 44.7%, brk 5.9%, delta_am +0.058, draft 0.762, final 0.820, **looks_good pass/fail 0.60/0.20** (fail doubled: more missed errors); oracle fix 48.9%, brk 5.4%, final 0.845 (refiner intact). kl_img ~0.004. checkpoint-50/100 kept. | `fir.md` |


## Plan (agreed 2026-10-12 by the user and jeet)
| Task | What | Cluster | Status |
|---|---|---|---|
| A | t11 (`darshan_t11_fir`) to step 300; evals every 50; stop-penalty decision with jeet if looks_good_fail keeps rising | Fir | running/queued |
| B | t8 ckpt-500 eval, full-history chains to depth 4 (`--eval_branch 1,1,1,1,1`, val + val_oracle per round) + SFT baseline | TamIA | to run |
| C | MDP eval mode in `rollout.py` (each round sees prompt + latest image + its critique only), then t8 at depth 4 | Fir writes, TamIA runs | to build |
| D | Official benchmarks (GenEval2, TIIF) of t8 merged-500 (+ rounds 1–4 if possible), then t11 ckpts; GenAI-bench setup | Mila | needs t8 copied to Mila |
| E | N=4 multi-round training rollout: chain to depth 4, 4 siblings at one random round (cost ~t8's), hybrid critiques, image tokens trained; full history or MDP per B/C | Fir writes | after B/C, design with jeet |
| F | N=4 training from t8 (merged base = anchor t8, or adapter on SFT: to decide) | Fir | after E and A |
| G | N=4 training from SFT (ablation) | TamIA | after E |
Open questions for jeet: t8 init t7 ckpt-200 (script) or 250 (message)? "3 chains" = 3 rounds in one chain? exact MDP
definition; N=4 tree shape/budget.

## Reference: the collaborator's runs (Fir, their account)
- `rl_ft_oracle_t7`: the recipe above at 36 prompts/step (2 nodes × 3 train ranks × 6, per its header;
  unconfirmed). The collaborator reports refined > draft. Their notes: oracle fix 15.3%, brk 7.0% at step 0;
  t5 (same idea, `ADV_NORM=mean`, KL 0.05, t20 policy) didn't learn (advantages ~0.02, KL penalty dominated).
- The collaborator's plan after t7 (as relayed by the user, 2026-10-11):
  - t8: t7 with different hyper-parameters (details unknown).
  - t9: train critique **and** refiner together (own critiques), started from a t7 checkpoint. **Failed**: without
    oracle critiques the refiner regressed. Reported "oracle critique +7.8 vs policy critique 3.8" (metric and step
    unknown; to ask). Don't repeat t9 as is.
  - t10: critique-only training (image tokens not trained). **Failed** (collaborator, 2026-10-11): training the critic
    drifted the refiner through the shared LoRA, and with image tokens out of the loss they were also out of the KL
    term, so nothing held them. (Also: the KL reference is the adapters-off model, i.e. SFT, not the t7 start.)
  - t11 (collaborator implementing; will push code + a launcher): merge the t8 LoRA into the SFT weights (so the
    adapters-off KL reference = t8), add a fresh LoRA, train critique tokens, and keep image tokens in the KL term.
    Things to check: separate pg / KL weights per token kind; normalize the critique pg loss over critique tokens so
    ~729 image KL tokens per refinement don't dilute it; an image-KL coefficient possibly > 0.05; watch
    `val_oracle/fix_1` and `val/am_0` for drift. Note `--image_token_weight 0` with `--critique_source hybrid` trains
    nothing (critiques are teacher-forced), as the Fir smoke test 64442582 confirmed: critique training needs
    `--critique_source policy`.
- `xiaofeng` branch: a different final-image GRPO design with a pixel judge; no clear gain by step 200.

## Cross-run findings
- Latent-reward RL improves the draft, not the refinement (`darshan_rl_2gpu_mila`).
- The refiner, not the critic, limits refinement: oracle critiques barely raise the fix rate
  (`darshan_rl_oracle_b25_mila` step 0; `darshan_t7_mila` step 0; collaborator's t5 probe).
- In oracle mode `val_oracle/am_1` averages only failing drafts (passing drafts get no child); compare
  `fix_1` / `am_final` instead.
- The t7 recipe does improve refinement where latent-reward RL did not: on Mila (18/step), by step 200 the val800
  fix rate with the model's own critique went 12.0% → 25.5%, final − draft −0.010 → +0.018, and final AM
  0.666 → 0.764 (drafts also improved, 0.676 → 0.746). With oracle critiques fix reaches 36.5%, so the critic now
  limits refinement more than at step 0 (attribute: 22.5% own vs 55% oracle). Count stays hardest (22.5% / 26%).
- The t7 recipe reproduces across clusters at 18 prompts/step: at step 100, `darshan_t7_fir` (H100) has
  own-critique final − draft +0.021, fix 24.5%, brk 7.2% vs `darshan_t7_mila` (A100) +0.013, 21.2%, 7.5%; oracle
  fix 28.9% vs 28.5%. Fir is slightly ahead on every metric (one run each, so seed noise is not ruled out).
- With the t7 recipe the refiner keeps improving but the critic stops keeping up: on Fir the own-critique fix rate is
  flat at ~24.5% from step 50 to 150 while the oracle-critique fix rate climbs 25% → 29% → 38%; Mila's step 200 shows
  the same gap (own 25.5% vs oracle 36.5%). The critique, not the refiner, now limits self-refinement.
- Larger batch is not ahead at equal steps so far: `darshan_t7_tamia` (36/step) at step 50 has own fix 18.7%,
  final − draft +0.001 vs `darshan_t7_fir` (18/step) 24.5%, +0.015 (one run each; early).
- Fir and Mila (same recipe, 18/step) track each other through step ~180 (wandb history: 20-step KL medians within
  ~0.05, train fix_1 and draft AM alike). At steps 181–200 Fir alone went unstable: KL median 0.52 (Mila 0.12),
  clip_frac 0.064 (0.002), train draft AM 0.690 (0.761), train reward_1 +0.086 (+0.040). Fir stayed in that regime
  to step 300 (KL medians 0.4–0.7). Step-200 val800: Mila draft 0.746 / final 0.764 / fix 25.5% vs Fir 0.684 /
  0.730 / 30.5%. So the draft drop is a run-specific instability, not a property of the recipe; in Fir's windows
  the refine reward rises exactly when the draft falls (181–200, 241–260), consistent with the refine reward
  (child − parent) rewarding weaker drafts. Best final AM so far: Mila checkpoint-200 (0.764), healthy draft;
  Fir checkpoint-300 0.762 with a weaker draft. Judge checkpoints by final AM, not by final − draft.
- t7 improves the refiner, not the critic (`critic_diag_*`, val800): the stop decision is unchanged after 200 steps
  (error recall ~92–94%, stop precision ~57–60%, stop recall ~48%: half of the drafts that already pass are refined
  anyway), and Corrections get narrower: with 2–3 failed objects the critic names all of them less often than SFT
  did (44% vs 55%, 22.5% vs 33%). When the critique does name a failed question, the fix rate doubles (12% → 26%).
  Targets for critic training (t11): name every failed object, and stop on drafts that pass.
- The t7 gain carries over to the official benchmarks (one seed) and grows with training: GenEval2 final AM 61.8
  (SFT t21) → 65.7 (ck-100) → 71.7 (ck-200), and refining now adds +2.9 AM there (SFT −0.2). TIIF short final 63.6 → 71.5. On TIIF long prompts the
  refinement lowers the score (63.2 → 61.7), so long captions are a weak spot of the refine step.
- H100 runs the t7 recipe at ~4.8 min/step (18 prompts/step) vs ~7.3–7.6 on A100.
