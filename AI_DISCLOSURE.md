# AI disclosure

Team Xeno (Aviral Malik, Vellore Institute of Technology, Vellore), Samsung XR Hackathon 2026, Track 3.

## AI tools used to build this entry

| Tool | What it was used for |
|---|---|
| Claude (Anthropic), through Claude Code | Literature research on inertial odometry; designing, writing and testing the estimator (`src/main.py`) and the earlier C++ version; running and logging the experiments (`docs/experiments.md`); test and measurement tools; the README and walkthrough; the presentation and the demo video script |
| OpenAI Codex | An independent, read-only review of the code with 50 extra test inputs. Its findings were checked by re-running them, and only fixes that held up on the official test files were kept |

## What the author did

Aviral Malik chose the track, set the direction and the constraints, decided which results and changes to keep, checked
the outputs, ran the demo and recorded the video.

## AI inside the submission

None. The submitted program (`src/main.py`) is plain Python and NumPy: no neural network, no learned weights, no
model files and no network access at run time. Every number it uses is either a physical constant, a fixed setting in
the code, or estimated from the input file itself.

## How results were checked

Every score in the README and the presentation was measured on files made with the organisers' own degradation script
(`degrade_imu.py`), using the scripts in `experiments/scripts/` and `tools/`. Nothing was tuned on ground truth from the
files being scored, and ideas that did not improve the scores are listed with their numbers in `docs/experiments.md`.
