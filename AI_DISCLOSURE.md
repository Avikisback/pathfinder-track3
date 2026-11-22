# AI disclosure

Team Xeno (Aviral Malik, Vellore Institute of Technology, Vellore), Samsung XR Hackathon 2026, Track 3.

## AI tools used to build this entry

| Tool | What it was used for |
|---|---|
| Claude (Anthropic), through Claude Code | Literature research on inertial odometry; designing, writing and testing the estimator (`src/main.py`) and the earlier C++ version; running and logging the experiments (`docs/experiments.md`); test and measurement tools; the README and walkthrough; the presentation and the demo video script |
| OpenAI Codex | An independent, read-only review of the code with 50 extra test inputs. Its findings were checked by re-running them, and only fixes that held up on the official test files were kept |

## What the author did

Aviral Malik led the project and made its decisions; the AI tools worked under his direction.

- **Problem and rules.** Chose Track 3, collected the organisers' material (brief, FAQ, degradation script and the
  submission template) and kept the entry to their rules, including the switch to a Python-only `main.py` after the
  FAQ of 17 November.
- **Direction.** Set the goals for each round of work (accuracy on the official degradation, memory and runtime limits,
  robustness to malformed input) and decided which results and changes went into the entry.
- **Independent review.** Brought in a second AI tool (Codex) as an outside reviewer of the code, and had every finding
  re-tested before anything was changed.
- **Checking.** Reviewed the presentation slide by slide and asked for corrections (names, diagram, layout, theme,
  consistency of the numbers), ran the entry and its checks himself, and confirmed the outputs.
- **Delivery.** Prepared the submission (repository, package, presentation), ran the live demo and recorded the video.

## AI inside the submission

None. The submitted program (`src/main.py`) is plain Python and NumPy: no neural network, no learned weights, no
model files and no network access at run time. Every number it uses is either a physical constant, a fixed setting in
the code, or estimated from the input file itself.

## How results were checked

Every score in the README and the presentation was measured on files made with the organisers' own degradation script
(`degrade_imu.py`), using the scripts in `experiments/scripts/` and `tools/`. Nothing was tuned on ground truth from the
files being scored, and ideas that did not improve the scores are listed with their numbers in `docs/experiments.md`.
