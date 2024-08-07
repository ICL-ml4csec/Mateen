# Rasd

<p align="center">
    <img width="600" src="https://github.com/ICL-ml4csec/Rasd/assets/62217808/6cdd4536-7461-402f-874c-a788efba0f8f" alt="Rasd">
</p>

# Overview
Rasd is a framework designed for semantic shift detection and adaptation in learning-based multi-class network intrusion detection systems. It comprises two main components:

### Shift Detection Function
  - Utilizes a centroid-based loss function to detect shifts.
  
### Shift Adaptation Module
  - Selects a representative subset of detected shift samples, approximating the entire distribution.
  - Manually labels this subset to train a Pseudo-Labeler, which is then used to pseudo-label the remaining unselected samples. 

For further details, please refer to the main paper.

# Pre-requisites and requirements
Before running Rasd, ensure you have the necessary dependencies installed. These dependencies are listed in the '<b>requirements.txt</b>' file. You can install them using the following command:
```bash
pip install -r requirements.txt
```

Here is the content of '<b>requirements.txt</b>':
```
torch==2.0.1
numpy==1.25.0
pandas==1.5.3
scipy==1.10.1
sklearn==1.2.2
deap==1.4
optuna==3.2.0
tqdm==4.65.0
```


# Models and Data 
You can download the pre-trained models and the processed data from the following link: 
<p align="center"> <a href="https://drive.google.com/drive/folders/1Cj6EhC9ydGhkg6wcBgpfqFSLvLLRuSqG?usp=sharing" target="_blank">Google Drive Folder</a> </p>

The contents of the download are as follows: 
- `RasdData.zip`: Contains the processed data.
- `RasdModels.zip`: Contains the pre-trained models.

Download and extract these files into the main directory of Rasd (i.e., `Rasd/`). This will ensure that the data and models are properly organized and ready for use.
 
# How to Use Rasd

To utilize Rasd with our settings, please follow these steps to set up the required datasets and run the framework.

## Dataset Setup

First, download the datasets as mentioned in the [Models and Data](https://github.com/ICL-ml4csec/Rasd/edit/main/README.md#models-and-data) section. Ensure that the files are organized in the following directories:

- `data/CICIDS2017/` for IDS2017
- `data/CICIDS2018/` for IDS2018

You can directly download and unzip the datasets into the main directory of Rasd (i.e., `Rasd/`).

## Running Rasd

To run Rasd, use the following command:

```bash
python Main.py
```
## Command-Line Options 
You can customize the execution using various command-line options:

### Dataset Selection
Switch between datasets using the '<b>--dataset_name</b>' option.

Example:
```bash
python Main.py --dataset_name "CICIDS2017"
```
<details>
  <summary>Options</summary>
   "CICIDS2017" and "CICIDS2018"
</details>

### Detection Threshold
Set the detection thresholds using the '<b>--acceptance_err</b>' option.

Example:
```bash
python Main.py --dataset_name "CICIDS2017" --acceptance_err 0.07
```
<details>
<summary>Options</summary>
0.01, 0.02, 0.03, 0.04, 0.05, 0.06,  0.07, 0.08,  0.09, and  0.1
</details>

### Training Mode
Use pre-trained models or train new models using the '<b>--train_mode</b>' option.

Example:
```bash
python Main.py --dataset_name "CICIDS2017" --acceptance_err 0.07 --train_mode "pre-train"
```
<details>
  <summary>Options</summary>
    "pre-train" and "train-new"
</details>

### Mode of Operation
Select the operation mode (detection only or detection & adaptation) using the '<b>--Mode</b>' option. 

Example:
```bash
python Main.py --dataset_name "CICIDS2017" --acceptance_err 0.07 --train_mode "pre-train" --Mode "Detection"
```
<details>
    <summary>Options</summary>
    "Detection" and "Both"
</details>

### Detection Method
Choose the detection method using the '<b>--Detection_Method</b>' option.

Example:
```bash
python Main.py --dataset_name "CICIDS2017" --acceptance_err 0.07 --train_mode "pre-train" --Mode "Detection" --Detection_Method "Rasd"
```
<details>
    <summary>Options</summary>
    "Rasd", "LSL", and "CADE"
</details>

### Selection Rate
Set the selection rate for building a subset for manual labeling using the '<b>--selection_rate</b>' option.

Example:
```bash
python Main.py --dataset_name "CICIDS2017" --acceptance_err 0.07 --train_mode "pre-train" --Mode "Detection" --Detection_Method "Rasd" --selection_rate 0.05
```
<details>
    <summary>Options</summary>
    0.01, 0.02, 0.03, 0.04, and 0.05
</details>

### Selection Batch Size
Set the batch size for splitting the pool of detected samples using the '<b>--selection_batch_size</b>' option.

Example:
```bash
python Main.py --dataset_name "CICIDS2017" --acceptance_err 0.07 --train_mode "pre-train" --Mode "Detection" --Detection_Method "Rasd" --selection_rate 0.05 --selection_batch_size 3000
```

<details>
    <summary>Options</summary>
    3000, 2000, and 1000
</details>

# Citation
```
@inproceedings{alotaibi24rasd,
  title={Rasd: Semantic Shift Detection and Adaptation for Network Intrusion Detection},
  author={Alotaibi, Fahad and Maffeis, Sergio},
  booktitle={the 39th International Conference on ICT Systems Security and Privacy Protection (SEC 2024)},
  pages={14},
  year={2024},
  organization={Springer}
}

```
# Contact

If you have any questions or need further assistance, please feel free to reach out to me at any time: 
- Email: `f.alotaibi21@imperial.ac.uk`
- Alternate Email: `fahadalkarshmi@gmail.com`
