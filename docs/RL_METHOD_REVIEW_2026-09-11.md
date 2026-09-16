# RL method review — September 11, 2026

Follow-up: both concerns now have opt-in flags; legacy defaults remain. See
`scripts/rl_ft/README.md` for usage and `docs/AGENT_HANDOFF.md` for validation.
The analysis below records the pre-flag implementation reviewed on this date.

Method-focused follow-up to `RL_REVIEW_2026-09-11.md`, reviewing the current
working tree. Production code and submitted jobs were not changed.

## Verdict

The implementation is a recognizable outcome-supervised GRPO variant: independent
episodes, same-prompt final-reward baselines, per-token clipped ratios, an SFT
reference, and per-episode token averaging. Mean-only advantages deliberately
omit the original paper's standard-deviation scaling. There is no confirmed
reward-sign, group-mixing, EOS-mask, token-shift, or gradient-accumulation error
in the tested default path.

However, matching original-style GRPO is not the same as having an unbiased
gradient of expected final-image reward. Two mathematical concerns affect the
current default implementation: response-length normalization and the gradient
of the sampled KL penalty. These are distinct from the operational findings in
the earlier review and from evidence about why a particular run's reward fell.

## 1. Episode-length normalization changes the reward gradient

Location: `llava/train/rl/rollout.py:319`.

With the default reflection weight of one, all sampled actions in episode i
receive weight `1 / L_i`, where L_i counts sampled semantic, reflection, EOS and
image-transition tokens. Forced tokens and padding do not count. The loss is
then averaged over episodes in `train_grpo.py:334` and `grpo.py:158`.

At the old policy, ignoring KL, the resulting ascent estimator is:

```
(1 / B) sum_i [(R_i - mean_group(R)) / L_i]
                   * sum_t grad log pi(a_it | history_it)
```

The score-function gradient of expected terminal reward instead uses the sum of
action log-probability gradients without the action-dependent `1 / L_i` factor.
A fixed denominator only changes scale; an episode's sampled length changes
relative weighting and can change direction. With independent group samples,
including the sample's own reward in a mean-only group baseline contributes the
usual `(G-1)/G` scale factor to the unnormalized expected reward gradient. That
constant is not the length-bias problem.

For the same positive advantage, an EOS or correction decision in an 800-token
episode receives four times the direct coefficient of one in a 3200-token
episode. For negative advantages, each action in a longer episode receives a
smaller penalty. This is relevant when choosing between stopping and generating
another 729-token image. It does not imply that every shorter trajectory wins:
actual parameter updates also depend on the token gradient vectors.

An adversarial probe exactly enumerated all 81 groups of four samples from a
three-outcome toy policy. Its outcome probabilities were all 1/3, final rewards
were `[0.9, 0.8, 0.1]`, lengths were `[800, 3200, 3200]`, and policy logits were
`theta * [1, -2, 1]`. Only the initial choice depends on theta; subsequent tokens
still count toward episode length. At theta=0:

- The true expected-reward derivative was approximately `-0.2`.
- Actual `build_training_rows` and `grpo_loss` produced expected ascent
  `+2.34375e-5`, a direction that decreases expected reward in this example.
- Replacing only the per-episode divisor with the constant 3200 produced ascent
  `-4.6875e-5`, agreeing with the reward-gradient direction.

This establishes a possible objective mismatch, not a diagnosis of the actual
run. Per-episode averaging is explicitly required by the current repository
instructions and matches the original GRPO normalization; it was not silently
changed. If the desired priority is expected final-image reward without this
length-dependent weighting, a Dr. GRPO-style constant denominator is the
methodological change to consider, with gradient scale/LR checked separately.

The original normalization appears in
[DeepSeekMath, Eq. 3](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS1).
The bias and constant-denominator alternative are analyzed in
[Understanding R1-Zero-Like Training, §3](https://arxiv.org/html/2503.20783v2#S3).

## 2. The KL estimator's value and gradient describe different divergences

Location: `llava/train/rl/grpo.py:154`.

The code computes `d = log(pi_ref) - log(pi)` and `k3 = exp(d) - d - 1`, then
backpropagates through k3 on fixed sampled tokens. For a fixed context with
on-policy actions, its expectation equals `KL(pi || pi_ref)`, as documented.
But differentiating only the fixed-sample expression omits the dependence of
the sampling distribution on pi. At that context, its expected gradient equals
the gradient of `KL(pi_ref || pi)` instead. With multiple optimizer epochs,
the sampling distribution additionally remains the old policy.

An exact three-action enumeration verified this using the production loss:

```
pi     = [0.7, 0.2, 0.1]
pi_ref = [0.2, 0.3, 0.5]

Expected production KL value: 0.6348972651
Exact KL(pi || pi_ref) value: 0.6348972651

Production logit gradient:             [ 0.500000, -0.100000, -0.400000]
Exact KL(pi || pi_ref) logit gradient:  [ 0.432506, -0.208072, -0.224434]
Exact KL(pi_ref || pi) logit gradient:  [ 0.500000, -0.100000, -0.400000]
```

Both divergences encourage agreement with the SFT reference, but weight
departures differently. This is a real discrepancy with the stated KL gradient,
not evidence that the KL term is inactive or that its sign is reversed.
Its size in the actual run has not been measured; beta is currently 0.01.

Multiplying k3 by the **differentiable** token importance ratio
`exp(logp - old_logp)` gives the intended conditional KL gradient in this probe.
The distinction still matters in the single-epoch case: the ratio's value is
one but its derivative is nonzero. Detaching that ratio would not fix the
on-policy gradient. Episode-length weighting is a separate issue; this
correction does not remove it or account for changing state visitation.

This inherited K3 issue and correction are described in
[DeepSeek-V3.2, §3.1, Eq. 7](https://arxiv.org/html/2512.02556v1#S3.SS1), and the
[TRL documentation](https://huggingface.co/docs/trl/grpo_trainer) documents the
differentiable ratio under `use_bias_correction_kl`. Production was not changed.

## What is implemented correctly

| Component | Evidence and interpretation |
|---|---|
| Independent episodes | `rollout.py:234` creates a separate initial draft for each sample; no shared draft or local correction tree. |
| Final-image reward | `train_grpo.py:267` renders only the last complete semantic image, then uses its Qwen AM as reward at line 278. |
| Same-prompt baseline | `train_grpo.py:282` groups by prompt index. `grpo.py:104` subtracts that group's mean. Default G=4. |
| Credit assignment | `rollout.py:316` assigns that single advantage to every sampled action, including the draft, subsequent images, reflection text and EOS. |
| Sampling distribution | Both phases use temperature 1, top-k 0, top-p 1. Image-only/text-only masks agree with the normalization in `grpo.py:39`. |
| Forced tokens and EOS | Forced prefixes/splices have no direct policy loss; sampled EOS and image transitions are retained. Padding is excluded independently of EOS IDs. |
| Autoregressive shift | `grpo.py:79` scores token t using the hidden state at t-1, with explicit position IDs and attention masks. |
| Old policy | `train_grpo.py:358` detaches current log-probabilities before any optimizer step, then retains them across PPO epochs. |
| PPO clipping | `grpo.py:150` has the correct sign-dependent minimum. Positive overlarge ratios and negative overly small ratios clip; the opposite sides retain gradients. |
| Gradient accumulation | All microbatches use the same episode-count denominator; the optimizer steps after accumulation and rank averaging. Existing microbatch/distributed tests passed. |
| Trainable parameters | Only first-model LoRA updates. The frozen output projection still transmits gradients into its inputs, allowing image and text probabilities to change. |

## Choices that affect learning but are not implementation mistakes

- **One optimizer epoch per rollout batch:** all old-policy ratios equal one
  before that update, so clipping is inactive. This is a legitimate on-policy
  configuration. It does not provide a bound on the size of the subsequent
  optimizer update. More epochs reuse samples but introduce stale-policy issues;
  increasing them is not an automatic fix.
- **Near-zero PG loss:** group-centered advantages and equal total weight per
  episode make the single-epoch PG value cancel. The gradient need not cancel.
  A direct probe returned loss zero and gradients `[-0.1, +0.1]`. Judge progress
  with fixed-prompt reward, policy change and gradient diagnostics.
- **Mean-only advantages:** consistent with the user's requested objective and
  avoids dividing small renderer reward differences by their small group std.
  It is a deliberate variant of original std-normalized GRPO.
- **Frozen stochastic renderer:** CFG in the second AR model is compatible with
  optimizing only the first model. Its random output is part of the environment;
  a differentiable renderer or a second-model policy loss is unnecessary. One
  rendering per episode adds noise to comparisons between four rollouts. Its
  variance has not been measured independently of first-model sampling.
- **Reflection is a means to reward:** final-image reward offers no explicit
  bonus for more turns or a more informative final reflection. If a draft is
  already rewarded highly, ending the episode can be successful behavior.
- **Caps define the trained task:** max refinements, reflection length or context
  can terminate without EOS. The last complete image still receives reward.
  Thus a high reward does not demonstrate correct EOS behavior. This matches the
  explicitly requested cap handling; it trains the capped process rather than
  an unlimited process guaranteed to end naturally.

## Validation and recommended next steps

The previous 28-test CPU suite passed in the actual Tar environment. Six new
method probes passed in that environment: exact variable-length counterexample;
exact KL-gradient comparison; four clipping sign/direction cases; zero-loss
nonzero-gradient case; whole-episode sampling/likelihood parity; and persistence
of old-policy snapshots across optimizer epochs.

The new likelihood check sampled 16 actual tiny-model episodes with nonzero
LoRA, variable prefixes, one to three images, forced reflection splices, EOS and
cap exits. All 202 sampled actions matched their recorded generation
log-probabilities to maximum absolute error `2.3841858e-7`. This is CPU float32
evidence; exact production FlashAttention-2/BF16 generation-versus-training
parity remains unmeasured. The production old-logprob recomputation makes the
initial ratio one by construction, so that ratio alone cannot establish parity.

Review evidence is in `results/reviews/rl-review-20260911/method_probes.json`;
the reproduction script is `method_probes.py` in the same directory.

Before another methodological comparison, prioritize deciding the length
normalization objective and correcting the conditional KL gradient if
`KL(pi || pi_ref)` is intended. Preserve final-pixel credit, phase masks and
mean-only group advantages. Validate any change against these probes, then
measure actual GPU likelihood parity and renderer reward variance. Neither the
probes nor the earlier logs establish which effect caused the observed reward
trend. No algorithm change or job restart was made during this review.
