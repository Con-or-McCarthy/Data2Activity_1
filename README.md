## Overview
This is the GitHub for the paper "[Forensic Activity Classification Using Digital Traces from iPhones: A Machine Learning-based Approach](https://arxiv.org/abs/2512.03786)" (accepted at DFRWS EU 2026).

This repo contains the code required to replicate the results from the paper,  namely training and testing a likelihood ratio (LR) system for classifying activities using Digital Trace data extracted from iPhones.

## Getting started

This repo uses Digital Trace data from [NFI\_FARED](https://huggingface.co/datasets/NetherlandsForensicInstitute/NFI_FARED_Digital_Traces). This is forensically available data extracted from iPhones. More information on the dataset can be found in our [paper](https://arxiv.org/abs/2512.03786). 

To get things going right away you can run the following commands in your terminal:
```bash
git clone https://github.com/Con-or-McCarthy/Data2Activity_1 
cd Data2Activity_1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash download_original_data.sh
python process_data.py
python main.py
```

This will clone the repo on your computer, install the required python packages to a virtual environment, download the NFI\_FARED dataset to `data/NFI_FARED/original`, process the data to `data/NFI_FARED/clean`, and run the code with default config parameters. Note: this has been verified for python=3.11.5 . 

## Use

This repo uses [weights and biases (wandb)](https://wandb.ai/home) to log results. It is recommended to make a wandb account and enter your information in `conf/wandb/wandb.yaml`. Otherwise results will only be printed to terminal and not stored anywhere.

The configuration of the runs can be adjusted directly in the `.yaml` files under the `conf/` folder, or as we recommend, in the command line when you run a script. For example, to train/test an LR system using a CatBoost scorer, Logistic Regression calibrator, with $H_1$=running and $H_2$=cycling, you would run:
```python
python main.py scorer=CatBoost calibrator=LogReg eval.activity_pair=['running','cycling']
```

Supported scorers are: `CatBoost`, `DecisionTree`, `RandomForest`, and `XGBoost`. Supported calibrators are: `Gaussian`, `KDE` (Kernel Density Estimation), and `LogReg` (Logistic Regression). Some of the hyperparameters of the scorers and calibrators can be further adjusted as desired. You can explore the config files for options. 

ECE, histogram, PAV, and Tippet plots are saved to the folder `figs/` at the end of each (binary) run. If wandb is available, they will also be saved online, as long as `eval.save_plots_wandb=True`. Be careful about saving too many figures to wandb since they can use a lot of your memory if performing a high number of runs. 

## MultiClass
Multiclass LR systems have also been implemented in this repo. This can be selected by specifying `eval.is_multiclass=True`. An example command is: 
```python 
python main.py scorer=CatBoost calibrator=LogReg eval.is_multiclass=True eval.expert_cluster_choices=['transport', 'movement', 'stationary']
```
To evaluate a multiclass scenario for the expert clusters of classes named 'transport', 'movement', and 'stationary'. Setting `eval.expert_cluster_choices=None` or not specifying a choice defaults to using all expert clusters. In the multiclass setting, metric $C_{llr}$ is replaced with the more general $C_{mxe}$. The reference value for $C_{mxe}$ ($log_2(K)$) as well as $\hat{C}_{mxe} = C_{mxe} / log_2(K)$ are both printed at the end of the run. 

Currently `Catboost` is the only scorer implemented for multiclass, and `LogReg` and `KDE` are the only calibrators.  

## Citation
If you wish to use this dataset in your research please cite:
```
@misc{mccarthy2025forensicactivityclassificationusing,
      title={Forensic Activity Classification Using Digital Traces from iPhones: A Machine Learning-based Approach}, 
      author={Conor McCarthy and Jan Peter van Zandwijk and Marcel Worring and Zeno Geradts},
      year={2025},
      eprint={2512.03786},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.03786}, 
}
```

## Contact
For questions regarding the data processing, the paper, and/or project GitHub please contact Conor McCarthy: c.t.mccarthy@uva.nl

For questions regarding the data collection please contact Jan Peter van Zandwijk: j.p.van.zandwijk@nfi.nl 
