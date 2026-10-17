# Lab notebook

One file per working day: `notes/YYYY-MM-DD.md`. Copy `_template.md`.

This is not a diary. It exists so that week 4 is assembly rather than
archaeology. The write-up needs an honest limitations section (brief 9) and a
method section that explains why each choice was made -- both are impossible to
reconstruct from memory three weeks after the fact, and neither lives in the
code.

## Tags

Write them inline, anywhere. They are just greppable strings.

| Tag | Use for | Feeds |
|---|---|---|
| `[DECISION]` | a choice made and why, including what you rejected | Method section |
| `[LIMITATION]` | something known-wrong, approximate, or unvalidated | Limitations section |
| `[OPEN]` | a question you could not answer and moved past | "What remains open" |
| `[SURPRISE]` | a result that contradicted your expectation | usually the most interesting paragraph in the report |

Harvest them at any point:

```bash
grep -rn "\[LIMITATION\]" notes/          # the limitations section, drafted
grep -rn "\[DECISION\]"   notes/          # every choice, with its date
grep -rn "\[SURPRISE\]"   notes/          # what you actually learned
```

## Rules that keep it useful

- **Write the negative results.** A parameter set that did nothing, a gradient
  that was garbage, an afternoon lost to a units bug -- these are the write-up.
  Brief 4.3.1 is explicit that honest mixed results build credibility.
- **Reference artifacts, not memory.** Commit SHA, config path, figure path
  under `reports/figures/`. "The Allan curve looked off" is useless in October;
  "adev at tau>100s ran 2x high, see figures/2026-09-20-adev-gyro.png, commit
  a3f21c" is a finding.
- **Numbers, not adjectives.** "Noisy" is not a measurement.
- **Five minutes at the end of a session beats an hour on Sunday.**
