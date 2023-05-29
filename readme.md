# MPhil ACS Project Explanation Generation

requirements.txt contains all dependencies

exp_train.py is the training scripts for explanation generation
multi_exp_gen.py is the inference script for generating explanations
prompt_tuning.py is the soft-prompt tuning training script
distribution_finetune.py is the training scripts for nli models to minimise KL divergence with extra explanations
nli_test.py is the inference scripts for nli classification
contrastive_eval.py is the scripts for contrastive token_level explanation generation
split_data.py is for data generation

Please check related config.py for hyperparameter settings

many ipynb for testing and contains small gadgets such as results post-processing
