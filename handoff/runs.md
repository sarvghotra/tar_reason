# RL runs (all clusters)

One row per run. Each session edits only the rows for its own cluster. Longer notes go in that cluster's
`<cluster>.md` Log. Status: `planned`, `queued`, `running`, `done`, `failed`, `cancelled`.

| Run (`RUN_NAME`) | Cluster | Recipe | Batch | Status | Key results | Detail |
|---|---|---|---|---|---|---|
| `darshan_rl_2gpu_mila` | Mila | latent reward (frozen Tar-7B), own critiques (trained), `BRANCH=4,2`, LR 1e-5 cosine, KL 0.01, 1 PPO epoch, `stop_penalty=0`, from t21 @17K, old data + val256 | 4/step (2× A100 × 2) | done 2026-10-07, 500 steps, job 11105481 | val256: draft AM 0.726 → 0.742 (peak 0.750 @400), final AM 0.720 → 0.745; final − draft −0.012…+0.003; fix ~12% throughout; KL ≤ 0.008. Drafting improved, refining didn't. | ckpts 450/475/500 in Mila `output_dir/darshan_rl_2gpu_mila/`; gallery https://claude.ai/artifact/RH2zJjLUTSsTtt6PKw3GhR |
| `darshan_rl_oracle_b25_mila` | Mila | latent reward, oracle critiques (pasted), `BRANCH=2,5`, `adv_std_floor 0.04`, untruncated image sampling, old data + val256 | 4/step (2× A100 × 2) | cancelled at step ~54, 2026-10-08 | step 0 val256: fix 16.3% own critique vs 17.1% oracle (by skill, oracle helps only attribute 20→38%; count 13→11%) → refiner is the bottleneck, not the critic | `mila.md` |
| `darshan_t7_mila` | Mila | t7: Qwen3-VL-8B pixel judge (GPU 3), hybrid critiques, `BRANCH=2,4`, LR 1e-4 const, KL 0.05, 2 PPO epochs, `draft_reward children`, stop 0.5@0.8, `DATASET_SEED=19`, data v2 + val800 | 18/step (3 A100 × 6) | done 2026-10-10 05:02, 200 steps (`MAX_STEPS=200`, evals at 0/100/200), jobs 11143217 + 11145380…91 | **val800, own critique (step 0 → 100 → 200):** draft AM 0.676 → 0.704 → **0.746**; final AM 0.666 → 0.718 → **0.764**; final − draft −0.010 → +0.013 → **+0.018**; fix 12.0% → 21.2% → **25.5%**; brk 8.9% → 7.5% → 7.1%; improved/degraded 16.5/21.9% → 22.5/16.6%; looks_good 12.2% → 15.8%. **Oracle critique:** final − draft +0.009 → +0.035 → **+0.056**; fix 14.9% → 28.5% → **36.5%**; brk 7.4% → 8.8% → 6.4%. Step-200 fix by skill, own/oracle: object 47/67%, count 22.5/26%, attribute 22.5/55%, position 26/33%, verb 15/20% (step 0: 16/18, 12/15, 8.5/14, 13/14, 7/5). One-step KL spikes at 13–14, 55 (3.5, grad 87), 186 (0.61), all recovered; KL baseline ~0.1–0.2 late. ~7.3 min/step. | `mila.md`; checkpoint-100 and -200 in Mila `$SCRATCH/tar_reason/kept/darshan_t7_mila/` |
| `t7_smoke_fir` | Fir | t7, 2 steps, eval on 24 prompts | 18/step (3 H100 × 6) | done 2026-10-09, job 63829085 | pass: judge 260 s, steps 0–2, ckpt 1.9 GB. Rollout ~170 s, reward ~81 s, train ~35 s = **~4.8 min/step** | `fir.md` |
| `darshan_t7_fir` | Fir | t7, same flags as `darshan_t7_mila`, 300 steps, val800 every 50 | 18/step (3 H100 × 6) | running since 2026-10-09 11:49 PDT; step 150 at 05:35 on 10-10 (job 63858144, resumed from ckpt-40 after a SIGBUS crash of 63858143 at step 41; spare 63858145 queued) | **val800, own critique** (step 0 → 50 → 100 → **150**): final − draft +0.002 → +0.015 → +0.021 → **+0.014**; fix 13.9% → 24.5% → 24.5% → **24.4%**; brk 7.7% → 9.9% → 7.2% → **7.3%**; draft AM 0.670 → 0.695 → 0.717 → **0.742**, final AM 0.672 → 0.710 → 0.738 → **0.756**; improved/degraded 22.3/16.4%, looks_good 12.9% → **18.5%**. **Oracle:** fix 14.5% → 25.1% → 28.9% → **38.2%**, brk 7.4% → 8.2% → 7.1% → **8.3%**, final AM **0.788**. KL ~0.1 at step 100, 0.2–0.4 in steps 145–167; one-step spike at 168 (1.37, grad 1.21), recovered next step. ~5.1 min/step; val800 eval ~80–95 min. | `fir.md`; checkpoint-100 kept in `~/scratch/tar_reason_kept/darshan_t7_fir/` |
| `t7_smoke_tamia` | TamIA | t7, 2 steps, eval on 24 prompts | 18/step (3 H100 × 6) | done 2026-10-09, job 515432 | pass: judge 40 s, steps 0–2, ckpt 1.9 GB, 21 min total. Rollout ~155 s, reward ~75 s, train 33–63 s = **~4.5–5 min/step** | `tamia.md` |
| `darshan_t7_tamia` | TamIA | t7, same flags as `darshan_t7_fir`, 300 steps, val800 every 50 | 36/step (2 nodes × 3 H100 × 6) | running since 2026-10-10 04:13 (jobs 516259 → 60 → 61, 12 h each) | **val800, own critique** (step 0 → **50**): final − draft +0.008 → **+0.001**; fix 14.9% → **18.7%**; brk 7.0% → **9.4%**; draft AM 0.675 → **0.696**, final AM 0.682 → **0.697**. **Oracle:** final − draft +0.011 → **+0.017**; fix 14.9% → **19.3%**, brk 7.7% → **8.3%**. Steps 1–50: KL 0 → 0.09 smooth, grad ≤ 0.06. Steps 67–71: multi-step wobble (KL peak 0.34, grad 0.67, clip_frac 0.28), recovered at 72. Spike at 88 (KL 1.63, grad 39), recovered by 90; ckpt-85 kept in `$SCRATCH/tar_reason_kept/darshan_t7_tamia/`. ~4.9 min/step at 36/step; val800 eval ~40 min. | `tamia.md` |

## Reference: the collaborator's runs (Fir, their account)
- `rl_ft_oracle_t7`: the recipe above at 36 prompts/step (2 nodes × 3 train ranks × 6, per its header;
  unconfirmed). The collaborator reports refined > draft. Their notes: oracle fix 15.3%, brk 7.0% at step 0;
  t5 (same idea, `ADV_NORM=mean`, KL 0.05, t20 policy) didn't learn (advantages ~0.02, KL penalty dominated).
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
- H100 runs the t7 recipe at ~4.8 min/step (18 prompts/step) vs ~7.3–7.6 on A100.
