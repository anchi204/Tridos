# Tridos

Implementation of **Triple-Domain Feature Learning With Frequency-Aware Memory Enhancement for Moving Infrared Small Target Detection**.

The implementation combines the Tridos spatial, temporal, and frequency-domain branches with a YOLOX-based detection backbone and head.

## Structure

```text
Tridos/
├── src/
│   ├── model.py
│   ├── dataset.py
│   └── config.py
├── train.py
├── evaluate.py
├── visualize.py
├── requirements.txt
└── README.md
```

The original Colab notebook was reorganized into these files without changing the model/training logic; the notebook remains the reference implementation used for the current run.

## Dataset

The implementation uses the ITSDT-15K dataset. Keep the dataset outside the repository.

Expected structure:

```text
ITSDT-15K/
├── annotations/
├── images/
└── splits.json
```

The current configuration uses:
- Input size: 512 × 512
- Temporal window: 5 frames
- Batch size: 4
- Mixed precision training

## Results from the current Colab run

| Confidence | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.05 | 41.0% | 58.4% | 48.2% |
| 0.20 | 47.3% | 56.2% | 51.4% |
| 0.40 | 51.8% | 54.1% | 53.0% |

## Dependencies

Install the Python dependencies from `requirements.txt`. YOLOX is used as the detection framework and is expected to be available on the Python path, as in the original Colab setup.

## Checkpoints

Model checkpoints and dataset files are intentionally excluded from Git. The training script saves `last.pth` and `best.pth` to the configured checkpoint directory.
