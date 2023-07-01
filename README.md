# TemplateGEC
This repository contains the source code for *TemplateGEC: Improving Grammatical Error Correction with Detection Template* accepted by ACL2023. This paper presents a new method for GEC, called TemplateGEC, which integrates the Seq2Edit and Seq2Seq frameworks, leveraging their strengths in error detection and correction.

## Overview

TemplateGEC utilizes the detection labels from a Seq2Edit model, to construct the template as the input. A Seq2Seq model is employed to enforce consistency between the predictions of different templates by utilizing consistency learning.

![1684048376086](image/1684048376086.jpg)

## Dataset

Download the datasets from [here](https://drive.google.com/file/d/15CkQmOOWuZJ344fEavCI0NtWEyelyGbH/view?usp=sharing) and unzip the `datas.zip` in the `scripts` directory.

## Train and Inference

**For `transformer` model:**

1. preprocess the raw data
   - `sh scripts/model_transformer/0_preprocess_baseline.sh` - for the baseline model
   - `sh scripts/model_transformer/0_preprocess_template-only.sh` - for the template-only model
   - `sh scripts/model_transformer/0_preprocess_template-consistency.sh` - for the template-consistency model
2. train the model
   - `sh scripts/model_transformer/1_train_transformer_baseline.sh` - for the baseline model
   - `sh scripts/model_transformer/1_train_transformer_template-only.sh`  - for the template-only model
   - `sh scripts/model_transformer/1_train_transformer_template-consistency.sh` - for the template-consistency model
3. inference
   - `sh scripts/model_transformer/2_generate_transformer_baseline.sh` - for the baseline model
   - `sh scripts/model_transformer/2_generate_transformer_template-only.sh`  - for the template-only model
   - `sh scripts/model_transformer/2_generate_transformer_template-consistency.sh` - for the template-consistency model

**For `bart` model:**

1. preprocess the raw data
   - `sh scripts/model_bart/0_preprocess_baseline.sh` - for the baseline model
   - `sh scripts/model_bart/0_preprocess_template-only.sh` - for the template-only model
   - the data-bin for the template-consistency model is already listed
2. train the model
   - `sh scripts/model_bart/1_train_bart_baseline.sh` - for the baseline model
   - `sh scripts/model_barty/1_train_bart_template-only.sh`  - for the template-only model
   - `sh scripts/model_bart/1_train_bart_template-consistency.sh` - for the template-consistency model
3. inference
   - `sh scripts/model_bart/2_generate_bart_baseline.sh` - for the baseline model
   - `sh scripts/model_bart/2_generate_bart_template-only.sh`  - for the template-only model
   - `sh scripts/model_bart/2_generate_bart_template-consistency.sh` - for the template-consistency model

**For `t5` model:**

1. train the model
   - `sh scripts/model_t5/1_train_t5_baseline.sh` - for the baseline model
   - `sh scripts/model_t5/1_train_t5_template-only.sh`  - for the template-only model
   - `sh scripts/model_t5/1_train_t5_template-consistency.sh` - for the template-consistency model
2. inference
   - `sh scripts/model_t5/2_generate_t5_baseline.sh` - for the baseline model
   - `sh scripts/model_t5/2_generate_t5_template-only.sh`  - for the template-only model
   - `sh scripts/model_t5/2_generate_t5_template-consistency.sh` - for the template-consistency model

## Evaluate

For bea-test dataset, please refer to [https://codalab.lisn.upsaclay.fr/competitions/4057](https://codalab.lisn.upsaclay.fr/competitions/4057).

For bea-valid and conll14-test datasets:

1. `cd scripts/metric`

3. `sh compute_conll14-test_m2.py` - evaluate conll14-test dataset

ACL2023: TemplateGEC: Improving Grammatical Error Correction with Detection Template
