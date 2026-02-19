# Fake News Detection with BERT

Fine-tuned BERT model for fake news classification (True/Fake) using news headlines, achieving 84% accuracy.
## Dataset
Download datasets here :
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download

True.csv and Fake.csv (44.8K samples, 2015-2017 US news)
## Features
- Binary classification on 44.8K samples (True.csv, Fake.csv)
- BERT tokenization (max_length=15) + custom classifier head
- Train/Val/Test split: 70%/15%/15%
- Performance: F1-score 0.84, Precision 0.85, Recall 0.84

## Quick Start
```bash
pip install transformers torch pandas numpy matplotlib scikit-learn
```
Run `FakeNewDetection.ipynb` in Jupyter.

## Results
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| True  | 0.77      | 0.93   | 0.84     |
| Fake  | 0.92      | 0.75   | 0.83     |

Model weights: `c2newmodelweights.pt`

## Limitations
- Trained on headlines only (2015-2017 US news)
- Limited hyperparameter tuning
