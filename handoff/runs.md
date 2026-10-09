# RL runs (all clusters)

One row per run. Each session edits only the rows for its own cluster. Longer notes go in that cluster's
`<cluster>.md` Log. Status: `planned`, `queued`, `running`, `done`, `failed`, `cancelled`.
Rows marked † were copied from the Fir copy of CLAUDE.md on 2026-10-09; the Mila session should verify them.

| Run (`RUN_NAME`) | Cluster | Recipe | Batch | Status | Key results | Detail |
|---|---|---|---|---|---|---|
| `darshan_rl_2gpu_mila` † | Mila | latent reward (frozen Tar-7B), own critiques, `BRANCH=4,2`, `stop_penalty=0`, from t21 @17K | 2× A100 | done 2026-10-07, 500 steps, job 11105481 | val256: draft AM 0.726 → 0.742 (peak 0.750 @400), final AM 0.720 → 0.745; final − draft −0.012…+0.003; fix ~12%. Drafting improved, refining didn't. | ckpts 450/475/500 in Mila `output_dir/darshan_rl_2gpu_mila/` |
| `darshan_rl_oracle_b25_mila` † | Mila | latent reward, oracle critiques, `BRANCH=2,5` | – | cancelled at step ~54, 2026-10-08 | step 0 val256: fix 16.3% own critique vs 17.1% oracle → refiner is the bottleneck, not the critic | `mila.md` |
| `darshan_t7_mila` † | Mila | t7: Qwen3-VL-8B pixel judge (GPU 3), hybrid critiques, `BRANCH=2,4`, LR 1e-4 const, KL 0.05, 2 PPO epochs, `draft_reward children`, stop 0.5@0.8, `DATASET_SEED=19`, data v2 + val800 | 18/step (3 A100 × 6) | running since 2026-10-08, `MAX_STEPS=200 EVAL_STEPS=100` (chain 11145380…91) | step 0 val800: own final−draft −0.010, fix 12.0%; oracle +0.009, fix 14.9%, brk 7.4%. Step 95: train fix_1 13.7% → 22.5%, fix_oracle_1 12.4% → 26%, reward_1 +0.01…+0.03, brk_1 ~7.5%. ~7.3 min/step. | `mila.md` |
| `t7_smoke_fir` | Fir | t7, 2 steps, eval on 24 prompts | 18/step (3 H100 × 6) | done 2026-10-09, job 63829085 | pass: judge 260 s, steps 0–2, ckpt 1.9 GB. Rollout ~170 s, reward ~81 s, train ~35 s = **~4.8 min/step** | `fir.md` |
| `darshan_t7_fir` | Fir | t7, same flags as `darshan_t7_mila`, 300 steps, val800 every 50 | 18/step (3 H100 × 6) | running since 2026-10-09 (jobs 63858143 → 44 → 45) | – | `fir.md` |
| `t7_smoke_tamia` | TamIA | t7, 2 steps, eval on 24 prompts | 18/step (3 H100 × 6) | done 2026-10-09, job 515432 | pass: judge 40 s, steps 0–2, ckpt 1.9 GB, 21 min total. Rollout ~155 s, reward ~75 s, train 33–63 s = **~4.5–5 min/step** | `tamia.md` |
| `darshan_t7_tamia` | TamIA | t7, same flags as `darshan_t7_fir`, 300 steps, val800 every 50 | 36/step (2 nodes × 3 H100 × 6) | queued 2026-10-09 (jobs 516259 → 60 → 61, 12 h each; est. start 23:45) | – | `tamia.md` |

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
- H100 runs the t7 recipe at ~4.8 min/step (18 prompts/step) vs ~7.3–7.6 on A100.
