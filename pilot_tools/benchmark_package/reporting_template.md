# Reporting Template

## Main principle

Do not report `ASR` alone.

Every main result table should report:

- `ACC`
- `ASR`
- `CACC`
- `Abstain`
- `Third-Answer Drift`

Add selective-behavior reporting:

- current operating-point coverage with explicit ASR threshold check
- risk-coverage frontier only when a real selection score sweep is available
- paired bootstrap confidence intervals for key deltas

## Failure taxonomy language

Use the same wording throughout the paper:

- `Gold Retention`
- `Targeted Hijack`
- `Third-Answer Drift`
- `Abstention`

## Recommended result layout

### Main benchmark table

Columns:

- `System`
- `Dataset`
- `ACC`
- `ASR`
- `CACC`
- `Abstain`
- `Third-Answer Drift`
- `current-op coverage (ASR<=0.10 check)`

Rows:

- `Clean-Strong`
- `Attacked Baseline`
- `Defended Baseline`

Scopes:

- `Combined`
- `Hotpot`
- `NQ`

### Clean ceiling ablations

Columns:

- `Retrieval Backend`
- `Hotpot Reader`
- `NQ Reader`
- `Context Style`
- `Combined ACC`
- `Hotpot ACC`
- `NQ ACC`

### Failure composition figure

Plot:

- stacked bar with `Gold Retention / Targeted Hijack / Third-Answer Drift / Abstain`

### Selective behavior table

Columns:

- `System`
- `Coverage`
- `ACC`
- `ASR`
- `CACC`
- `current-op coverage (ASR<=0.10 check)`
- optional `frontier coverage@ASR<=0.10` when score sweeps are available

### Significance appendix

Report paired bootstrap deltas for:

- `ACC`
- `ASR`
- `CACC`
- `Abstain`
- `Third-Answer Drift`

## Writing guidance

When describing attacks:

- mention the drop in gold retention
- mention the increase in third-answer drift
- mention ASR as one component, not the sole summary

When describing defenses:

- mention ASR reduction
- mention abstention increase
- mention `CACC` to characterize answered-subset recovery

## Canonicalization policy

- Headline tables should use the official evaluator's conservative canonicalization, including sentence-final standalone `Unknown` suffixes mapped to `Unknown` / abstain.
- If desired, report suffix-stripped recovery only as a sensitivity analysis in the appendix, never as the headline benchmark number.

## Example interpretation template

Attack:

- The attack does not only increase targeted hijack; it also substantially reduces gold retention and induces large third-answer drift.

Defense:

- The defense lowers targeted hijack but pays for this gain with reduced coverage, making the security-coverage trade-off explicit.
