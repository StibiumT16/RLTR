# Reinforcement Learning To Rank Using Coarse-grained Rewards

### Data Preparation
We follow the pipeline in [ULTRA](https://github.com/ULTR-Community/ULTRA_pytorch/tree/main/example) to preprocess the datasets.

We also provide processed datasets. You can download them to `./data` via this [Google Drive Link](https://drive.google.com/drive/folders/1zIQNiuSFeUTTKIvkoSZOYnqlEAOT5KPM?usp=drive_link)


### Model Training
First set the model's config in `./config`. We provide an example config of GRPO. 

After setting the model parameters, we can start training the model. For example, to train a GRPO model on the Yahoo dataset for 10,000 steps:
```bash
python src/main.py -o train \
    --input_feed deterministic_online_label_input \
    --config_path config/grpo.yaml \
    --data_path dataset/yahoo \
    --log_path log/yahoo_grpo \
    --model_save_path model/yahoo_grpo.pt \
    -l 0.0001 -s 20000 
```

In `example_lr_grid_search.sh`, we provide a script for performing a grid search on the learning rate.


### Model Evaluation
For example, to evaluate the GRPO model on the Yahoo dataset, the command is as follows:
```bash
python src/main.py -o test \
    --config_path config/grpo.yaml \
    --data_path dataset/yahoo \
    --model_save_path model/yahoo_grpo.pt \
    --output_path output/yahoo_grpo.csv
```

The `output/istella` directory contains test results on the Istella dataset using an MLP (DNN) as the backbone. The number at the end of each filename represents the optimal learning rate obtained through grid search.