# Mateen

<p align="center">
    <img width="600" src="https://github.com/ICL-ml4csec/Rasd/assets/62217808/6cdd4536-7461-402f-874c-a788efba0f8f" alt="Rasd">
</p>

# Overview
Mateen is an ensemble framework designed to enhance AutoEncoder (AE)-based one-class network intrusion detection systems by effectively managing distribution shifts in network traffic. It comprises four key components:

### Shift Detection Function
  - **Functionality**: Uses statistical methods to detect shifts in the distribution of network traffic data.
  
### Sample Selection
  - **Subset Selection**: Identifies a representative subset of the network traffic samples that reflects the overall distribution after a shift.
  - **Labeling and Update Decision**: The subset is manually labeled to decide whether an update to the ensemble is necessary.

### Shift Adaptation Module
  - **Incremental Model Update**: Integrates the benign data of the labeled subset with the existing training set. Then, updates the incremental model on this expanded set. 
  - **Temporary Model Training**: Initiates a new temporary model with the same weights as the incremental model. Then, train this model exclusively on the benign data of the labeled subset.
    
### Complexity Reduction Module
  - **Model Merging**: Combines temporary models with similar performance.
  - **Model Pruning**: Discards underperforming models.

For further details, please refer to the main paper.

# Pre-requisites and requirements
Before running Mateen, ensure you have the necessary dependencies installed. These dependencies are listed in the '<b>requirements.txt</b>' file. You can install them using the following command:
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
tqdm==4.65.0
```


# Models and Data 
You can download the pre-trained models, the processed data, as well as the results CSV files from the following link: 
<p align="center"> <a href="https://drive.google.com/drive/folders/1PG_tPCxmS2rdkIMokjBnQkXhIJgJJlEY?usp=drive_link" target="_blank">Google Drive Folder</a> </p>

The contents of the folder are as follows: 
- `Datasets.zip`: Contains the processed data.
- `Models.zip`: Contains the pre-trained models.
- `Results.zip`: Contains prediction results and probability scores from the Mateen framework across the datasets

Download and extract these files into the main directory of Mateen (i.e., `Mateen/`). This will ensure that the data and models are properly organized and ready for use.
 
# How to Use Mateen

To utilize Mateen with our settings, please follow these steps to set up the required datasets and run the framework.

## Dataset Setup

First, download the datasets as mentioned in the [Models and Data](https://github.com/ICL-ml4csec/Mateen/edit/main/README.md#models-and-data) section. Ensure that the files are organized in the following directories:

- `Datasets/CICIDS2017/` for IDS2017
- `Datasets/IDS2018/` for IDS2018
- `Datasets/Kitsune/` for Kitsune and its variants. 

You can directly download and unzip the datasets into the main directory of Mateen (i.e., `Mateen/`).

## Running Mateen

To run Mateen, use the following command:

```bash
python Mateen.py
```
## Command-Line Options 
You can customize the execution using various command-line options:

### Dataset Selection
Switch between datasets using the '<b>--dataset_name</b>' option.

Example:
```bash
python Main.py --dataset_name "IDS2017"
```
<details>
  <summary>Options</summary>
   "IDS2017", "IDS2018", "Kitsune", "mKitsune", and "rKitsune"
</details>

### Window Size
Set the window size using the '<b>--window_size</b>' option.

Example:
```bash
python Main.py --dataset_name "IDS2017" --window_size 50000
```
<details>
<summary>Options</summary>
10000, 50000, and 100000
</details>

### Performance Threshold
The minimum acceptable performance '<b>--performance_thres</b>' option.

Example:
```bash
python Main.py --dataset_name "IDS2017" --window_size 50000 --performance_thres 0.99
```
<details>
  <summary>Options</summary>
    0.99, 0.95, 0.90, 0.85, and 0.8
</details>

### Maximum Ensemble Size
The maximum acceptable ensemble size '<b>--max_ensemble_length</b>' option. 

Example:
```bash
python Main.py --dataset_name "IDS2017" --window_size 50000 --performance_thres 0.99 --max_ensemble_length 3
```
<details>
    <summary>Options</summary>
    3, 5, and 7
</details>

### Selection Rate
Set the selection rate for building a subset for manual labeling using the '<b>--selection_budget</b>' option.

Example:
```bash
python Main.py  --dataset_name "IDS2017" --window_size 50000 --performance_thres 0.99 --max_ensemble_length 3 --selection_budget 0.01
```
<details>
    <summary>Options</summary>
   0.005, 0.01, 0.05, and 0.1
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
@inproceedings{alotaibi24mateen,
  title={Mateen: Adaptive Ensemble Learning for Network Anomaly Detection},
  author={Alotaibi, Fahad and Maffeis, Sergio},
  booktitle={the 27th International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2024)},
  year={2024},
  organization={Association for Computing Machinery}
}

```
# Contact

If you have any questions or need further assistance, please feel free to reach out to me at any time: 
- Email: `f.alotaibi21@imperial.ac.uk`
- Alternate Email: `fahadalkarshmi@gmail.com`
