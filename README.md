# Overview
This is the GitHub for the paper "[Forensic Activity Classification Using Digital Traces from iPhones: A Machine Learning-based Approach](https://arxiv.org/abs/2512.03786)" (accepted at DFRWS EU 2026).

This repo contains the code required to replicate the results from the paper,  namely training and testing a likelihood ratio (LR) system for classifying activities using Digital Trace data extracted from iPhones.

## Getting started

This repo uses Digital Trace data from [NFI-FARED](https://huggingface.co/datasets/NetherlandsForensicInstitute/NFI_FARED_Digital_Traces). This is forensically available data extracted from iPhones. More information on the dataset can be found in our [paper](https://arxiv.org/abs/2512.03786). 

To get things going right away you can run the following commands in your terminal:
```bash
git clone https://github.com/Con-or-McCarthy/Data2Activity_1 
cd Data2Activity_1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash download_original_data.sh
python process_data.py
```

This will clone the repo on your computer, install the required python packages to a virtual environment, download the NFI-FARED dataset to `data/NFI_FARED/original` and process the data to `data/NFI_FARED/clean`. Note: this has been verified for python=3.11.5 . 

# Get results on your own data

If you have extracted your own set of Digital Traces from an iPhone, you can use this repo directly to calculate likelihood ratios of activities in the binary and multiclass cases.

First, you must process the extracted traces into `.pkl` files following the steps below:
==<<Jan Peter writes instructions to process extracted traces to .pkl files>>==

Once processed, your `/user_data/` folder should look like this:
```
user_data/
      ├── pkl_files/
            ├──df_dict_Cache.pkl
            ├──df_dict_healthdb_distance.pkl
            ├──df_dict_healthdb_floors.pkl
            ├──df_dict_healthdb_steps.pkl
            ├──df_dict_motionstate.pkl
            ├──df_dict_natalie.pkl
            └──df_dict_stepcounthistory.pkl
      ├── processed/
      └── output/
```
### Option 1: Binary
If you wish to get the likelihood ratio between two activities at each timestamp, you can simply run the command:
```
python use_your_data.py eval.activity_pair=['<activity_0>','<activity_1>']
```
Replacing `activity_0` and `activity_1` with your desired activities. This will process your `.pkl` files into a combined `.csv` file stored in `/user_data/processed/`, then train the model on all of NFI-FARED and produce likelihood ratios. The output will be stored in `user_data/output/output.csv` and will look like this:
```
timestamp,          | <activity_0>/<activity_1>  | <activity_1>/<activity_0>
2026-06-08 10:01:00 | 0.04293277541592836        | 23.292228147658793
2026-06-08 10:02:00 | 0.04293277541592836        | 23.292228147658793
2026-06-08 10:03:00 | 0.0315379409092649         | 31.707840498433747

```
This is compatible with the other configuration options available in `/conf/` (some more information below). For example, if you wish to run analysis for the activities "running" and "car", using only data from an Iphone6+ (iOS 11.4.1) carried in the back or front pocket, you can run the command:
```
python use_your_data.pt eval.activity_pair=['running','car'] eval.phone_types=['Iphone6+_IOS_11.4.1'] eval.carry_location=['backpocket', 'frontpocket']
```
You can explore the configuration files for more information on the possible commands.

### Option 2: Multiclass
The script also works for multiclass analysis. You need only to specify `eval.is_multiclass=True` in the command line. Setting `eval.expert_cluster_choices=null` or else not specifying will use all expert clusters. You may select a relevant subset (recommended) like so:
```
python use_your_data.py eval.is_multiclass=True eval.expert_cluster_choices=['movement', 'dynamic', 'stationary']
```
This uses only the listed expert clusters. The output of this command would look like this:
```
timestamp           | movement              | dynamic             | stationary
2026-06-08 10:01:00 | 0.0059647064080381096 | 0.03388746796117619 | 0.9601478256307858
2026-06-08 10:02:00 | 0.0059647064080381096 | 0.03388746796117619 | 0.9601478256307858
2026-06-08 10:03:00 | 0.02355575445898262   | 0.08651708433174253 | 0.8899271612092748
```
and be stored in `/user_data/output/output.csv`. Note that the numbers are the likelihoods, not the LRs. We leave it up to you to decide how to calculate the LR from the raw likelihood. Multiclass similarly works with other configuration specifications. Please consult the paper (*Results/Multiclass LR Systems*) for information on the validity of different cluster combinations.

By default, `use_your_data.py` uses the subfolders of `/user_data/`, you may specify your own `.pkl` folder and output folder like so:
```
python use_your_data.py +pkl_path="mypickles/pickles" +output_path="results/my_output.csv"
```



# Replicate the paper's results

If you simply wish to replicate the results in the paper, you do not need to upload your own data and may run analysis using NFI-FARED. Instructions below.

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
To evaluate a multiclass scenario for the expert clusters of classes named 'transport', 'movement', and 'stationary'. Setting eval.expert_cluster_choices=None or not specifying a choice defaults to using all expert clusters. In the multiclass setting, metric $C_{llr}$ is replaced with the more general $C_{mxe}$. The reference value for $C_{mxe}$ ($log_2(K)$) as well as $\hat{C}_{mxe} = C_{mxe} / log_2(K)$ are both printed at the end of the run.

Currently `Catboost` is the only scorer implemented for multiclass, and `LogReg` and `KDE` are the only calibrators.  

# Citation
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

# Contact
For questions regarding the data processing, the paper, and/or project GitHub please contact Conor McCarthy: c.t.mccarthy@uva.nl

For questions regarding the data collection please contact Jan Peter van Zandwijk: j.p.van.zandwijk@nfi.nl 
