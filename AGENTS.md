# AGENTS.md

## Project
This project is an edge-AI respiratory health monitoring prototype using microphone audio and LD6002 radar.

Main training route:
1. MIC first: cough vs non_cough baseline.
2. Radar second: parsed radar features, not raw UART bytes.
3. Fusion third: rule-based state machine first, small MLP later.
4. Teacher/student distillation only after baseline is stable.

## Data rules
- Never modify or delete files under data/raw/.
- Never commit raw audio, raw radar data, processed features, trained models, or results.
- All train/val/test splits must be grouped by person_id.
- Do not randomly split by clip_id because that causes subject leakage.
- Every generated artifact must be reproducible from configs/*.yaml.
- Every training run must save:
  - config copy
  - metrics.json
  - confusion_matrix.png
  - classification_report.txt
  - best model checkpoint

## Audio baseline
- Task: cough / non_cough.
- Sample rate: 16000 Hz.
- Window length: 1.0 s.
- Hop length: 0.5 s.
- Feature: 40-bin Log-Mel.
- First model: DS-CNN or small CNN.
- Metrics: accuracy, precision, recall, F1, confusion matrix.
- For cough class, prioritize recall and F1.

## Radar baseline
- Input is parsed radar CSV, not UART raw bytes.
- Expected columns may include timestamp, rr, hr, presence_score, motion_score, phase_amp, phase_var.
- Window length: 5 s or 10 s.
- First model: Random Forest or XGBoost.
- Also output feature importance.

## Fusion
- First version should be rule-based.
- Inputs:
  - audio cough probability
  - radar presence/motion/respiration quality
  - signal quality scores
- Outputs:
  - normal
  - attention
  - warning
- Small MLP is allowed only after rule baseline is working.

## Verification
Before saying a task is done, run:
1. Dataset check on a tiny sample.
2. Preprocessing smoke test.
3. One-epoch training smoke test.
4. Evaluation script.
5. Save results under results/.

## Coding style
- Use Python 3.10+.
- Use argparse for all scripts.
- Add clear error messages.
- Prefer simple, readable code over clever abstractions.
- Keep functions small and testable.
