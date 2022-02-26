# Apples-to-Apples: Comparing the Performance of Hate Speech Detection Models in Context

**Context**: Project for [CS6471](https://www.cc.gatech.edu/classes/AY2022/cs6471_spring/) course at Georgia Tech, Spring 2022.

**Authors**: 
- Seema Baddam
- Richard Huang
- Kai McKeever

## Installation phase

Please refer to [install.md](docs/install.md).

## Datasets: Offensive Language Identification Dataset - OLID 

[Dataset Paper](https://arxiv.org/abs/1902.09666) |
[Dataset Link1](https://scholar.harvard.edu/malmasi/olid) |
[Dataset Link2](https://sites.google.com/site/offensevalsharedtask/offenseval2019)

For this study, we use the sub-task A of the OLID Dataset. This dataset contains English tweets annotated using a three-level annotation hierarchy and was used in the OffensEval challenge in 2019. 

Preprocessing functions can be found in [preprocess_utils.py](src/utils/preprocess_utils.py).

## Preprocessing phase

Before attempting the training phase, please use this command to preprocess the data:

```bash
### Start preprocessing | Default to all dataset
python -m src.utils.preprocess_utils --dataset_name all
```

## Training phase

Please refer to [training.md](docs/training.md).

