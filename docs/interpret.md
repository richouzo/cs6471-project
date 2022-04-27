# XAI Interpretability instructions

⚠️ **DISCLAIMER: This part of the study contains words or language that are considered profane, vulgar, or offensive by some readers.** ⚠️

[Main README](../README.md)

All commands should be used from root directory.

## Retrieve stats for visualisation and attribution scores on all datasets

Stats csv files are saved in `stats-results/` folder, run this command before running the notebooks:

```bash
# DistillBert on OffensEval dataset
python -m src.evaluation.test_save_stats --test_dataset_name offenseval --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-15_02-48-34_trained_testAcc=0.8026.pth --loss_criterion crossentropy --only_test 0 --stats_label 1

# DistillBert on Implicit Hate dataset
python -m src.evaluation.test_save_stats --test_dataset_name implicithate --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-18_02-48-16_trained_testAcc=0.7585.pth --loss_criterion crossentropy --only_test 0 --stats_label 1

# DistillBert on Covid Hate dataset
python -m src.evaluation.test_save_stats --test_dataset_name covidhate --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-18_02-24-40_trained_testAcc=0.8397.pth --loss_criterion crossentropy --only_test 0 --stats_label 1
```

## Retrieve stats for Cross-domains on OffensEval

Stats csv files are saved in the `stats-results/` folder, run this command before running the notebooks:

```bash
# DistillBert on OffensEval dataset
python -m src.evaluation.test_save_stats --test_dataset_name offenseval --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-15_02-48-34_trained_testAcc=0.8026.pth --loss_criterion crossentropy --only_test 0 --stats_label 1

# DistillBert on Implicit Hate dataset
python -m src.evaluation.test_save_stats --test_dataset_name offenseval --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-18_02-48-16_trained_testAcc=0.7585.pth --loss_criterion crossentropy --only_test 0 --stats_label 1

# DistillBert on Covid Hate dataset
python -m src.evaluation.test_save_stats --test_dataset_name offenseval --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-18_02-24-40_trained_testAcc=0.8397.pth --loss_criterion crossentropy --only_test 0 --stats_label 1
```

## Word importance visualisations with Captum:

This is the qualitative part of our XAI results. We provide two notebooks to visualize which parts of the input sentence are used for an inference of a trained model.

In the current state, we use Integrated Gradients from [Captum](https://captum.ai/) library to obtain the attribution scores for each word in a given sentence. 

- For CNN/RNN-based models, please use this [XAI LSTM notebook](../src/evaluation/explainability_visualization.ipynb) (Example on our best BasicLSTM trained model).

- For BERT-based models, please use this [XAI Bert notebook](../src/evaluation/explainability_visualization_bert.ipynb) (Example on our best DistillBert trained model).

## Attribution scores statistics:

This is the quantitative part of our XAI results. We provide a notebook on the attribution scores statistics, with word clouds and distribution plots of the attribution scores. 

Attribution scores stats pkl files are saved in the `stats-results/` folder.

Please refer to [Attribution scores notebook](../src/evaluation/attribution_scores_stats.ipynb).

## Word importance and word cloud examples on OffensEval

#### Some samples from OffensEval with true positive predictions
![DistillBert_TP](docs/assets/DistillBert_TP.png)

#### Highest attribution scores words for OffensEval
![WordCloud_offenseval](docs/assets/high_attrib_wordcloud_2022-04-15_02-48-34_offenseval_offenseval.png)
