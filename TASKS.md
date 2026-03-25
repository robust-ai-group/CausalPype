# CausalPype Tasks Reference

## Estimation

### ATE (Average Treatment Effect)
Answers: "What is the average causal effect of setting treatment to value A vs value B on the outcome?"
Uses `gcm.average_causal_effect` under the hood. Generates interventional samples from the SCM for both treatment and control values, compares the mean outcome.

### CATE (Conditional Average Treatment Effect)
Answers: "Does the treatment effect vary across subgroups defined by effect modifiers?"
Wraps EconML estimators (LinearDML, CausalForestDML, TLearner). Auto-computes the backdoor adjustment set from the DAG via d-separation so the user doesn't have to specify confounders manually.

### KNNIntervention (K-Nearest Neighbor Matching)
Answers: "What is the treatment effect estimated by matching each unit to similar units with opposite treatment?"
Propensity-free. Matches treated to K nearest controls (and vice versa) on covariates. Reports ATE, ATT, ATC, and match quality. Does not use the SCM — pure matching on the data.

## Counterfactual / Interventional

### Counterfactual
Answers: "For these specific observed individuals, what would have happened if we changed X?"
Uses `gcm.counterfactual_samples` which inverts noise terms from observed data then replays the SCM under intervention. Individual-level, not population-level.

### Intervention
Answers: "What is the expected distribution of outcomes under do(X=x)?"
Uses `gcm.interventional_samples`. Population-level (generates new samples), not individual-level. Supports multi-node interventions.

### StochasticIntervention
Answers: "What happens if we nudge the treatment distribution rather than hard-set it?"
For continuous treatments: adds a shift to the natural value. For binary: flips 0→1 with some probability. Compares shifted vs baseline distributions.

### DoseResponse
Answers: "How does E[Y] change as we sweep treatment across a range of values?"
Runs `gcm.interventional_samples` at each treatment value in a grid. Returns a DataFrame of (treatment_value, expected_outcome, std).

## Mechanistic Understanding

### Mediation
Answers: "How much of the total effect goes through mediators (indirect) vs directly?"
Computes Natural Direct Effect (NDE) and Natural Indirect Effect (NIE) using Pearl's nested counterfactual formulas. Uses `_IndexedIntervention` to fix mediators at their individual-level counterfactual values.

### ArrowStrength
Answers: "How strong is each edge pointing into this target node?"
Uses `gcm.arrow_strength`. Quantifies per-edge causal strength (how much removing that edge changes the target's distribution).

### IntrinsicCausalInfluence
Answers: "What fraction of the target's variance is attributable to each upstream node's noise?"
Uses `gcm.intrinsic_causal_influence` (Shapley-based variance decomposition). Returns both raw and normalized (percentage) contributions.

## Anomaly & Distribution

### AnomalyAttribution
Answers: "For anomalous observations of the target, which upstream nodes caused the anomaly?"
Uses `gcm.anomaly_scores` to auto-detect anomalies (or accepts user-provided anomaly data), then `gcm.attribute_anomalies` to decompose the anomaly score.

### DistributionChange
Answers: "The target's distribution shifted between old_data and new_data — which nodes are responsible?"
Uses `gcm.distribution_change`. Decomposes the distributional shift into per-node contributions. Note: model should be fitted on old_data for correct results.

## Fairness & Validation

### FairnessAudit
Answers: "Would outcomes change if the protected attribute had been different, for each observed individual?"
Counterfactual fairness. Computes counterfactual outcomes under privileged vs unprivileged values. Reports disparity, observational gap, and per-individual unfairness.

### Validate
Answers: "Are the model's causal assumptions consistent with the data?"
Runs `gcm.refute_causal_structure` (edge dependence + local Markov tests) and `gcm.refute_invertible_model` (noise independence). Reports pass/fail with p-values.
