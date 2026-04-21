# CausalPype — Simpson's Paradox Demo
#
# Example of using CausalPype on a sythetic example of Simpson's Paradox.

from pathlib import Path
import numpy as np
import pandas as pd

import causalpype as cp

FIGDIR = Path('figures/simpson'); FIGDIR.mkdir(parents=True, exist_ok=True)
np.random.seed(42)


# ## 1. Generate Synthetic Data
#
# A hospital treats patients with varying disease severity. Patients with higher
# severity are more likely to receive an experimental treatment (confounding by
# indication). The treatment helps, but correlation analysis suggests it hurts
# because treated patients have worse outcomes on average.

print("=" * 60)
print("Simpson's Paradox Demo: Treatment Effect in a Hospital Setting")
print("=" * 60)

n = 3000

# Disease severity (confounder)
severity = np.random.normal(50, 25, n).clip(10, 90)

# Age affects severity and outcomes
age = np.random.normal(55, 12, n).clip(25, 85)
severity = severity + 0.3 * (age - 55)

# Sicker and older patients more likely to get experimental treatment (logistic model)
treatment_prob = 1 / (1 + np.exp(-(0.05 * (severity - 50) + 0.02 * (age - 55))))
treatment = (np.random.random(n) < treatment_prob).astype(int)

# Outcome: recovery score (0-100, higher is better)
# - Severity hurts recovery (strong negative effect)
# - Treatment helps recovery (positive causal effect)
# - Age has small negative effect
recovery = (
    70
    - 0.6 * (severity - 50)      # severity hurts
    + 8.0 * treatment            # treatment helps
    - 0.1 * (age - 55)           # age slightly hurts
    + np.random.normal(0, 8, n)   # noise
).clip(0, 100)

df = pd.DataFrame({
    'age': age,
    'severity': severity,
    'treatment': treatment,
    'recovery': recovery,
})

print("\nData Summary:")
print(f"  Patients: {n}")
print(f"  Treated: {treatment.sum()} ({100*treatment.mean():.1f}%)")
print(f"  Mean recovery (treated):   {recovery[treatment==1].mean():.1f}")
print(f"  Mean recovery (untreated): {recovery[treatment==0].mean():.1f}")

# Naive correlation
naive_effect = recovery[treatment==1].mean() - recovery[treatment==0].mean()
print(f"\n  NAIVE EFFECT: {naive_effect:+.2f} points")
print(" This suggests treatment hurts patients!")
print(" However this is not accurate. It is a case of Simpson's Paradox.")


# ## 2. Define the Causal DAG
#
# Severity is a confounder. It causes both treatment assignment and recovery.
# Without adjusting for it, we get a biased estimate.

print("\n" + "=" * 60)
print("Causal Analysis with CausalPype")
print("=" * 60)

dag = {
    'age':       ['severity', 'treatment', 'recovery'],
    'severity':  ['treatment', 'recovery'],
    'treatment': ['recovery'],
}

model = cp.CausalModel(dag, assignment_quality='better')

# Fit the model to data (learns the structural equations).
model.fit(df)

fig, _ = cp.plotting.plot_graph(model, title="Simpson's Paradox: Causal DAG")
fig.savefig(FIGDIR / 'dag.pdf')
print("\nDAG saved to figures/simpson/dag.pdf")

## 3. Estimate the True Causal Effect

print("\nEstimating Average Treatment Effect...")

r_ate = model.run(cp.ATE(
    treatment='treatment',
    outcome='recovery',
    treatment_value=1,
    control_value=0,
    num_samples=5000,
))
print(r_ate)

fig, _ = cp.plotting.plot_effects([r_ate], title="Treatment Effect on Recovery")
fig.savefig(FIGDIR / 'ate.pdf')

print(f"\n  TRUE CAUSAL EFFECT: {r_ate.estimate:+.2f} points")
print(f"  (Ground truth: +8.0 points)")
print(f"\n  Naive effect was {naive_effect:+.2f}, causal effect is {r_ate.estimate:+.2f}")

## 4. Mechanistic Drivers
#
# What contributes most to recovery outcomes?

print("\nAnalyzing mechanistic drivers...")

r_arrow, r_ici = model.run([
    cp.ArrowStrength(target='recovery'),
    cp.IntrinsicCausalInfluence(target='recovery'),
])
print(r_arrow)
print(r_ici)
fig, _ = cp.plotting.plot_arrow_strength(r_arrow, normalize=True)
fig.savefig(FIGDIR / 'arrow_strength.pdf')
fig, _ = cp.plotting.plot_influences(r_ici)
fig.savefig(FIGDIR / 'influences.pdf')

## 5. Heterogeneous Treatment Effects
#
# Does the treatment work better for some patients than others?

print("\nEstimating heterogeneous effects (CATE)...")

r_cate = model.run(cp.CATE(
    treatment='treatment',
    outcome='recovery',
    effect_modifiers=['age', 'severity'],
    method='causal_forest',
    n_estimators=200,
    random_state=42,
))
print(r_cate)

fig, _ = cp.plotting.plot_cate_distribution(
    r_cate,
    data=df,
    covariate='severity',
    covariate_label='Disease Severity',
    title='Treatment Effect by Severity'
)
fig.savefig(FIGDIR / 'cate_by_severity.pdf')


## 6. Counterfactual: What If We Treated Everyone?

print("\nCounterfactual: Universal treatment vs. no treatment...")

r_all_treat, r_no_treat = model.run([
    cp.Intervention(interventions={'treatment': 1}, outcome='recovery', num_samples=5000),
    cp.Intervention(interventions={'treatment': 0}, outcome='recovery', num_samples=5000),
])

print(f"  E[recovery | do(treatment=1)]: {r_all_treat.estimate:.2f}")
print(f"  E[recovery | do(treatment=0)]: {r_no_treat.estimate:.2f}")
print(f"  Causal effect: {r_all_treat.estimate - r_no_treat.estimate:+.2f}")


## 7. Dose-Response: Effect Across Severity Levels
#
# Using causal effect curves to show treatment benefit at different severity levels.

print("\nDose-response: Recovery across severity levels...")

r_curve = model.run(cp.CausalEffectCurve(
    treatment='severity',
    outcome='recovery',
    n_points=12,
    num_samples=3000,
))

fig, _ = cp.plotting.plot_causal_effect_curve(
    r_curve,
    title='Recovery vs Disease Severity'
)
fig.savefig(FIGDIR / 'dose_response.pdf')


## 8. Sensitivity Analysis
#
# How robust is our causal estimate to model misspecification?

print("\nSensitivity analysis...")

r_sens = model.run(cp.SensitivityAnalysis(
    treatment='treatment',
    outcome='recovery',
    methods=['placebo', 'subset', 'random_common_cause'],
    num_simulations=5,
    num_samples=2000,
))
print(r_sens)

fig, _ = cp.plotting.plot_sensitivity(r_sens)
fig.savefig(FIGDIR / 'sensitivity.pdf')


## 9. Summary
print("\n" + "=" * 60)
print("SUMMARY: Simpson's Paradox")
print("=" * 60)
print(f"""
In this demo, we saw:

1. NAIVE ANALYSIS: Treatment appears to HURT patients ({naive_effect:+.1f} points)
   - Treated patients have worse outcomes on average

2. CAUSAL ANALYSIS: Treatment actually HELPS patients ({r_ate.estimate:+.1f} points)
   - Severity is a confounder
   - After adjustment, the true beneficial effect is recovered
   - This matches the ground truth (+8.0 points)

3. SIMPSON'S PARADOX:
   - Sicker patients are more likely to receive treatment
   - Sicker patients have worse outcomes regardless of treatment
   - This creates a spurious negative correlation
   - Applying causal inference can untangle this
""")
