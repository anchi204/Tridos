# Tridos

Implementation of **Triple-Domain Feature Learning With Frequency-Aware Memory Enhancement for Moving Infrared Small Target Detection**.

The project uses the Tridos spatial, temporal, and frequency-domain feature branches with a YOLOX-based detection backbone and head.

## Structure

```text
Tridos/
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── requirements.txt
└── README.md
```

## Dataset

The implementation uses the ITSDT-15K dataset. Keep the dataset outside the repository and pass its path to the training/evaluation scripts.

Expected structure:

```text
ITSDT-15K/
├── annotations/
├── images/
└── splits.json
```

## Results from the current Colab run

| Confidence | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.05 | 41.0% | 58.4% | 48.2% |
| 0.20 | 47.3% | 56.2% | 51.4% |
| 0.40 | 51.8% | 54.1% | 53.0% |

## Notes

- Input size: 512 × 512
- Temporal window: 5 frames
- Training uses mixed precision and checkpoint resume/save.
- YOLOX is used as the detection framework.
