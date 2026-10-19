# Day 10: DoWhy Causal Audit

## Objective
Audit the synthetic retail treatment-effect estimate using DoWhy and
compare the estimate with the known synthetic ground truth.

## Results
- DoWhy linear-regression ATE: 0.079692
- sklearn regression-adjustment ATE: 0.079692
- Known synthetic ATE: 0.054113
- Absolute estimation error: 0.025579

## Refutation Tests

### Random Common Cause
- Original estimate: 0.079692
- Refuted estimate: 0.079693
- p-value: 0.96

The estimate changed very little after adding a random common cause.
This supports stability under this particular refutation test, but does
not prove that the causal assumptions are valid.

### Placebo Treatment
- Original estimate: 0.079692
- Placebo estimate: approximately 0
- p-value: 0.0

The placebo treatment produced an effect close to zero, consistent with
the intended placebo check. The p-value should be interpreted in the
context of the refutation procedure and is not, by itself, proof of
causality.

## Limitations
- The data is synthetic; results do not establish real-world pricing effects.
- Refutation tests cannot prove that all confounding has been removed.
- The ATE estimate differs from the known synthetic ground truth.
- DoWhy 0.8 requires a local compatibility patch for pandas 3.
- No warnings were recorded in the generated report; this does not
  establish that every possible warning has been eliminated.

## Conclusion
The DoWhy audit and both refutation procedures completed. The results
provide diagnostic evidence for this synthetic experiment, not proof
of causal identification or real-world effectiveness.
