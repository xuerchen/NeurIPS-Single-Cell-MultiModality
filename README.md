# NeurIPS-Single-Cell-MultiModality
This repo contains our solution to the [OpenProblems-NeurIPS2021 Single-Cell Multimodal Data Integration](https://eval.ai/web/challenges/challenge-page/1111/overview). Our team AXX took the [4th place of the modality prediction task](https://eval.ai/web/challenges/challenge-page/1111/leaderboard/2860) in terms of overall ranking of 4 subtasks: namely `GEX to ADT`, `ADT to GEX`, `GEX to ATAC` and `ATAC to GEX`. More details can be found in the [competition webpage](https://openproblems.bio/neurips_docs/about_tasks/task1_modality_prediction/). Our scripts are mostly the python files and the rest of the repo is copied from the [competition starter kit](https://github.com/openproblems-bio/neurips2021_multimodal_viash/releases/download/1.4.0/starter_kit-predict_modality-python.zip). 


### System requirement
- `NVIDIA GPU with 8 GB memory`
- `Linux OS` such as `Ubuntu`

### Install packages
- `pip install -r requirements.txt`
- install [Viash dependencies](https://openproblems.bio/neurips_docs/submission/quickstart/#2-configure-your-local-environment). 

### Download datasets
- from the root of this repo, `cd scripts`
- `./2_generate_submission.sh`

Please ignore the failure message of running the script since we haven't train the models yet and please ignore/delete the output `submission_phase1v2.zip`. The purpose of running this script at this point is two folds:
1. build the docker image.
2. download the input datasets

### Train the models
- `python train.py`
This script will train models for all 4 subtasks.

### Prediction and sanity check
- from the root of this repo, `cd scripts`
- `./2_generate_submission.sh`
- `./3_evaluate_submission.sh`

### Generate the final submission
- from the root of this repo, `cd scripts`
- `./4_generate_phase2_submission.sh`