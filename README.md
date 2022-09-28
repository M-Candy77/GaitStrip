# GaitStrip

The instructions is based on [GaitSet](https://github.com/AbnerHqC/GaitSet).    
## Dataset & Preparation

Before training or test, please make sure you have prepared the dataset by this two steps:

* Step1: Organize the directory as: `your_dataset_path/subject_ids/walking_conditions/views`. E.g.` OUMVLP/00001/00/000/.`
* Step2: Cut and align the raw silhouettes with `pretreatment.py`. 
## Pretreatment

Pretreatment your dataset by

        python pretreatment.py --input_path='root_path_of_raw_dataset' --output_path='root_path_for_output'
* `--input_path`**(NECESSARY)** Root path of raw dataset.

* `--output_path`**(NECESSARY)** Root path of raw output.

* `--log_file`Log file path. #Default: './pretreatment.log'

* `--log`If set as True, all logs will be saved. Otherwise, only warnings and errors will be saved. #Default: False

* `--worker_num` How many subprocesses to use for data pretreatment. Default: 1

## Configuration

In `config.py`, you might want to change the following settings:

* `dataset_path`**(NECESSARY)** root path of the dataset (for the above example, it is "gaitdata")

* `WORK_PATH`path to save/load checkpoints

* `CUDA_VISIBLE_DEVICES`indices of GPUs

## Train

Train a model by
        
        python train.py


## Evaluation
Evaluate the trained model by

        python testall.py
