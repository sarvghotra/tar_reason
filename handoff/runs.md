# RL runs (all clusters)

One row per run. Each session edits only the rows for its own cluster. Longer notes go in that cluster's
`<cluster>.md` Log. Status: `planned`, `queued`, `running`, `done`, `failed`, `cancelled`.

| Run (`RUN_NAME`) | Cluster | Recipe | Batch | Status | Key results | Detail |
|---|---|---|---|---|---|---|
| `darshan_rl_2gpu_mila` | Mila | latent reward (frozen Tar-7B), own critiques (trained), `BRANCH=4,2`, LR 1e-5 cosine, KL 0.01, 1 PPO epoch, `stop_penalty=0`, from t21 @17K, old data + val256 | 4/step (2× A100 × 2) | done 2026-10-07, 500 steps, job 11105481 | val256: draft AM 0.726 → 0.742 (peak 0.750 @400), final AM 0.720 → 0.745; final − draft −0.012…+0.003; fix ~12% throughout; KL ≤ 0.008. Drafting improved, refining didn't. | ckpts 450/475/500 in Mila `output_dir/darshan_rl_2gpu_mila/`; gallery https://claude.ai/artifact/RH2zJjLUTSsTtt6PKw3GhR |
| `darshan_rl_oracle_b25_mila` | Mila | latent reward, oracle critiques (pasted), `BRANCH=2,5`, `adv_std_floor 0.04`, untruncated image sampling, old data + val256 | 4/step (2× A100 × 2) | cancelled at step ~54, 2026-10-08 | step 0 val256: fix 16.3% own critique vs 17.1% oracle (by skill, oracle helps only attribute 20→38%; count 13→11%) → refiner is the bottleneck, not the critic | `mila.md` |
| `darshan_t7_mila` | Mila | t7: Qwen3-VL-8B pixel judge (GPU 3), hybrid critiques, `BRANCH=2,4`, LR 1e-4 const, KL 0.05, 2 PPO epochs, `draft_reward children`, stop 0.5@0.8, `DATASET_SEED=19`, data v2 + val800 | 18/step (3 A100 × 6) | running since 2026-10-08; at step ~193 on 2026-10-10 02:30 (step-200 eval due ~05:30 EDT); `MAX_STEPS=200 EVAL_STEPS=100` (chain 11145380…91) | **val800, own critique:** final − draft −0.010 (step 0) → **+0.013** (step 100); fix 12.0% → **21.2%**, brk 8.9% → 7.5%; draft AM 0.676 → 0.704, final AM 0.666 → 0.718. **Oracle critique:** final − draft +0.009 → **+0.035**; fix 14.9% → **28.5%**, brk 7.4% → 8.8%. Train steps 96–104: fix_1 25%, reward_1 +0.036. One-step KL spikes at 13–14 (1.2), 55 (3.5, grad 87) and 186 (0.61), all recovered; KL baseline ~0.1–0.2 by steps 180–193. ~7.3 min/step. | `mila.md`; checkpoint-100 kept in `$SCRATCH/tar_reason/kept/darshan_t7_mila/` |
| `t7_smoke_fir` | Fir | t7, 2 steps, eval on 24 prompts | 18/step (3 H100 × 6) | done 2026-10-09, job 63829085 | pass: judge 260 s, steps 0–2, ckpt 1.9 GB. Rollout ~170 s, reward ~81 s, train ~35 s = **~4.8 min/step** | `fir.md` |
| `darshan_t7_fir` | Fir | t7, same flags as `darshan_t7_mila`, 300 steps, val800 every 50 | 18/step (3 H100 × 6) | running since 2026-10-09 11:49 PDT; step ~95 at 23:30 (job 63858144, resumed from ckpt-40 after a SIGBUS crash of 63858143 at step 41; spare 63858145 queued) | **val800, own critique:** final − draft +0.002 (step 0) → **+0.015** (step 50); fix 13.9% → **24.5%**, brk 7.7% → 9.9%; draft AM 0.670 → 0.695, final AM 0.672 → 0.710. **Oracle:** fix 14.5% → 25.1%, brk 7.4% → 8.2%. Train steps 68–95: fix_1 22.9%, brk_1 7.0%, reward_1 +0.030, KL ~0.085 (one-step spike 0.35, grad 1.8, in steps 41–67). ~5.1 min/step; val800 eval ~80 min. | `fir.md` |
| `t7_smoke_tamia` | TamIA | t7, 2 steps, eval on 24 prompts | 18/step (3 H100 × 6) | done 2026-10-09, job 515432 | pass: judge 40 s, steps 0–2, ckpt 1.9 GB, 21 min total. Rollout ~155 s, reward ~75 s, train 33–63 s = **~4.5–5 min/step** | `tamia.md` |
| `darshan_t7_tamia` | TamIA | t7, same flags as `darshan_t7_fir`, 300 steps, val800 every 50 | 36/step (2 nodes × 3 H100 × 6) | running since 2026-10-10 04:13 (jobs 516259 → 60 → 61, 12 h each) | – | `tamia.md` |

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
- The t7 recipe does improve refinement where latent-reward RL did not: on Mila (18/step) val800 fix rate with the
  model's own critique went 12.0% → 21.2% and final − draft −0.010 → +0.013 by step 100 (`darshan_t7_mila`).
- `darshan_t7_fir` (H100, same recipe and batch as `darshan_t7_mila`) reached a similar refinement gain by step 50
  (own-critique fix 24.5%, final − draft +0.015) as Mila did by step 100 (21.2%, +0.013); one run each, so
  seed/sampling noise is not ruled out.
- H100 runs the t7 recipe at ~4.8 min/step (18 prompts/step) vs ~7.3–7.6 on A100.
