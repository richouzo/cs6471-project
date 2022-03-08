# 🍎 Apples-to-Apples: Comparing the Performance of Hate Speech Detection Models in Context

**Context**: Project for [CS6471](https://www.cc.gatech.edu/classes/AY2022/cs6471_spring/) course at Georgia Tech, Spring 2022.

**Authors**: 
- Seema Baddam
- Richard Huang
- Kai McKeever

## Installation phase

Please refer to [install.md](docs/install.md).

## Datasets

**Datasets used**:
- Offensive Language Identification Dataset
- Implicit Hate Speech Dataset
- Racism is a Virus Dataset
- Social Bias Frames Dataset

Please refer to [datasets.md](docs/datasets.md) for more details.

## Preprocessing phase

Before attempting the training phase, please use this command to preprocess the data:

```bash
### Start preprocessing | Default to all dataset
python -m src.utils.preprocess_utils --dataset_name all
```

## Training phase

Please refer to [training.md](docs/training.md).

