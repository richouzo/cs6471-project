# Evaluation instructions

[Main README](../README.md)

All commands should be used from root directory.

## Cross-domains evaluation

Evaluate models on different domain datasets:

```bash
# Evaluate BasicLSTM trained on OffensEval dataset
python -m src.evaluation.test_save_stats --test_dataset_name offenseval --model BasicLSTM --saved_model_path saved-models/BasicLSTM_2022-03-07_18-08-17_trained_testAcc=0.7155.pth --loss_criterion bcelosswithlogits --only_test 1

python -m src.evaluation.test_save_stats --test_dataset_name implicithate --model BasicLSTM --saved_model_path saved-models/BasicLSTM_2022-03-07_18-08-17_trained_testAcc=0.7155.pth --loss_criterion bcelosswithlogits --only_test 1 --original_dataset offenseval

python -m src.evaluation.test_save_stats --test_dataset_name covidhate --model BasicLSTM --saved_model_path saved-models/BasicLSTM_2022-03-07_18-08-17_trained_testAcc=0.7155.pth --loss_criterion bcelosswithlogits --only_test 1 --original_dataset offenseval


# Evaluate BasicLSTM trained on Implicit Hate dataset
python -m src.evaluation.test_save_stats --test_dataset_name offenseval --model BasicLSTM --saved_model_path saved-models/BasicLSTM_2022-03-08_00-40-58_trained_testAcc=0.7107.pth --loss_criterion bcelosswithlogits --only_test 1 --original_dataset implicithate

python -m src.evaluation.test_save_stats --test_dataset_name implicithate --model BasicLSTM --saved_model_path saved-models/BasicLSTM_2022-03-08_00-40-58_trained_testAcc=0.7107.pth --loss_criterion bcelosswithlogits --only_test 1

python -m src.evaluation.test_save_stats --test_dataset_name covidhate --model BasicLSTM --saved_model_path saved-models/BasicLSTM_2022-03-08_00-40-58_trained_testAcc=0.7107.pth --loss_criterion bcelosswithlogits --only_test 1 --original_dataset implicithate


# Evaluate BasicLSTM trained on Covid Hate dataset
python -m src.evaluation.test_save_stats --test_dataset_name offenseval --model BasicLSTM --saved_model_path saved-models/BasicLSTM_2022-03-08_00-37-58_trained_testAcc=0.7278.pth --loss_criterion bcelosswithlogits --only_test 1 --original_dataset covidhate

python -m src.evaluation.test_save_stats --test_dataset_name implicithate --model BasicLSTM --saved_model_path saved-models/BasicLSTM_2022-03-08_00-37-58_trained_testAcc=0.7278.pth --loss_criterion bcelosswithlogits --only_test 1 --original_dataset covidhate

python -m src.evaluation.test_save_stats --test_dataset_name covidhate --model BasicLSTM --saved_model_path saved-models/BasicLSTM_2022-03-08_00-37-58_trained_testAcc=0.7278.pth --loss_criterion bcelosswithlogits --only_test 1
```

```bash
# Evaluate DistillBert trained on OffensEval dataset
python -m src.evaluation.test_save_stats --test_dataset_name offenseval --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-15_02-48-34_trained_testAcc=0.8026.pth --loss_criterion crossentropy --only_test 1

python -m src.evaluation.test_save_stats --test_dataset_name implicithate --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-15_02-48-34_trained_testAcc=0.8026.pth --loss_criterion crossentropy --only_test 1

python -m src.evaluation.test_save_stats --test_dataset_name covidhate --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-15_02-48-34_trained_testAcc=0.8026.pth --loss_criterion crossentropy --only_test 1


# Evaluate DistillBert trained on Implicit Hate dataset
python -m src.evaluation.test_save_stats --test_dataset_name offenseval --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-18_02-48-16_trained_testAcc=0.7585.pth --loss_criterion crossentropy --only_test 1

python -m src.evaluation.test_save_stats --test_dataset_name implicithate --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-18_02-48-16_trained_testAcc=0.7585.pth --loss_criterion crossentropy --only_test 1

python -m src.evaluation.test_save_stats --test_dataset_name covidhate --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-18_02-48-16_trained_testAcc=0.7585.pth --loss_criterion crossentropy --only_test 1


# Evaluate DistillBert trained on Covid Hate dataset
python -m src.evaluation.test_save_stats --test_dataset_name offenseval --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-18_02-24-40_trained_testAcc=0.8397.pth --loss_criterion crossentropy --only_test 1

python -m src.evaluation.test_save_stats --test_dataset_name implicithate --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-18_02-24-40_trained_testAcc=0.8397.pth --loss_criterion crossentropy --only_test 1

python -m src.evaluation.test_save_stats --test_dataset_name covidhate --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-18_02-24-40_trained_testAcc=0.8397.pth --loss_criterion crossentropy --only_test 1
```
