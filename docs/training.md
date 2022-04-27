# Training instructions

[Main README](../README.md)

All commands should be run from root directory.

## Training

Figures and saved models are saved after training in their respective folder.

```bash
### Start training | Default to BasicLSTM
python -m src.training.main
```


## Gridsearch training

Gridsearch csv files are saved in `gridsearch-results/` folder.

You can modify the gridsearch parameters in [gridsearch_config.yml](../gridsearch_config.yml) before running this command:

```bash
### Start gridsearch
python -m src.training.gridsearch
```

| Hyperparameters      | Possible values |
| ----------- | ----------- |
| model_type  | ['BasicLSTM', 'DistillBert', 'DistillBertEmotion']       |
| optimizer_type   | ['adam', 'adamw', 'sgd']        |
| loss_criterion   | ['bceloss', 'bcelosswithlogits', 'crossentropy']        |
| lr   | [*float*]        |
| epochs   | [*int*]        |
| batch_size   | [*int*]        |
| patience_es   | [*int*]        |
| scheduler_type   | ['', reduce_lr_on_plateau', <br />'linear_schedule_with_warmup']        |
| patience_lr   | [*int*]        |
| save_condition   | ['loss', 'acc']        |
| fix_length   | [*null* or *int*]        |

*Note: 
- For BasicLSTM, please use the loss 'bcelosswithlogits'. 
- For 'DistillBert' and 'DistillBertEmotion', please use the loss 'crossentropy'*

## Testing

To print the model size parameters, loss and accuracy on the test set and save the confusion matrix run this command:

```bash
### Example on the BasicLSTM model trained on OffensEval and evaluated on OffensEval
python -m src.evaluation.test_save_stats --test_dataset_name offenseval --model BasicLSTM --saved_model_path saved-models/BasicLSTM_2022-03-07_18-08-17_trained_testAcc=0.7155.pth --loss_criterion bcelosswithlogits --only_test 1
```

```bash
### Example on the DistillBert model trained on OffensEval and evaluated on OffensEval
python -m src.evaluation.test_save_stats --test_dataset_name offenseval --model DistillBert --saved_model_path saved-models/DistillBert_2022-04-15_02-48-34_trained_testAcc=0.8026.pth --loss_criterion crossentropy --only_test 1
```

More details on evaluation (and Cross-domain evaluation) can be found in [evaluation.md](../docs/evaluation.md).
