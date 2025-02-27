
<p align="center">
  <img width="587" alt="mtn" src="https://github.com/user-attachments/assets/63a75b73-840d-4d62-8bed-34c157b869de">
</p>


# Overview
Mateen is an ensemble framework designed to enhance AutoEncoder (AE)-based one-class network intrusion detection systems by effectively managing distribution shifts in network traffic. It comprises four key components:

### Shift Detection Function
  - **Purpose**: Detects distribution shifts in network traffic using statistical methods.
  
### Sample Selection
  - **Subset Selection**: Identifies a representative subset of the network traffic samples that reflects the overall distribution after a shift.
  - **Labeling and Update Decision**: The subset is manually labeled to decide whether an update to the ensemble is necessary.

### Shift Adaptation Module
  - **Incremental Model Update**: Integrates the benign data of the labeled subset with the existing training set. Then, updates the incremental model on this expanded set. 
  - **Temporary Model Training**: Initiates a new temporary model with the same weights as the incremental model. Then, train this model exclusively on the benign data of the labeled subset.
    
### Complexity Reduction Module
  - **Model Merging**: Merges temporary models that perform similarly.
  - **Model Pruning**: Removes models that underperform compared to the best-performing model.

For further details, please refer to the main paper.

# Pre-requisites and requirements
Ensure the following dependencies are installed before running Mateen. You can install them using the command below:
```bash
pip install -r requirements.txt
```

Contents of '<b>requirements.txt</b>':
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
- `Results.zip`: Prediction results and probabilities across datasets.

Ensure these files are placed in the `Mateen/` directory after downloading and extracting.

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
python Mateen.py --dataset_name "IDS2017"
```
<details>
  <summary>Options</summary>
   "IDS2017", "IDS2018", "Kitsune", "mKitsune", and "rKitsune"
</details>

### Window Size
Set the window size using the '<b>--window_size</b>' option.

Example:
```bash
python Mateen.py --dataset_name "IDS2017" --window_size 50000
```
<details>
<summary>Options</summary>
10000, 50000, and 100000
</details>

### Shift Detection Threshold
Set the threshold using '<b>--shift_threshold</b>' option.

Example:
```bash
python Mateen.py  --dataset_name "IDS2017" --window_size 50000 --shift_threshold 0.05
```

<details>
    <summary>Options</summary>
    0.05, 0.1, and 0.2
</details>

### Performance Threshold
The minimum acceptable performance '<b>--performance_thres</b>' option.

Example:
```bash
python Mateen.py --dataset_name "IDS2017" --window_size 50000 --shift_threshold 0.05 --performance_thres 0.99
```
<details>
  <summary>Options</summary>
    0.99, 0.95, 0.90, 0.85, and 0.8
</details>

### Maximum Ensemble Size
The maximum acceptable ensemble size '<b>--max_ensemble_length</b>' option. 

Example:
```bash
python Mateen.py --dataset_name "IDS2017" --window_size 50000 --shift_threshold 0.05 --performance_thres 0.99 --max_ensemble_length 3
```
<details>
    <summary>Options</summary>
    3, 5, and 7
</details>

### Selection Rate
Set the selection rate for building a subset for manual labeling using the '<b>--selection_budget</b>' option.

Example:
```bash
python Mateen.py  --dataset_name "IDS2017" --window_size 50000 --shift_threshold 0.05 --performance_thres 0.99 --max_ensemble_length 3 --selection_budget 0.01
```
<details>
    <summary>Options</summary>
   0.005, 0.01, 0.05, and 0.1
</details>

### Mini Batch Size for Sample Selection
Choose the min-batch size using the '<b>--mini_batch_size</b>' option.

Example:
```bash
python Mateen.py --dataset_name "IDS2017" --window_size 50000 --shift_threshold 0.05 --performance_thres 0.99 --max_ensemble_length 3 --selection_budget 0.01 --mini_batch_size 1000
```
<details>
    <summary>Options</summary>
    500, 1000, and 1500
</details>


### Retention Rate
Set the value of the retention rate using '<b>--retention_rate</b>' option.

Example:
```bash
python Mateen.py --dataset_name "IDS2017" --window_size 50000 --shift_threshold 0.05 --performance_thres 0.99 --max_ensemble_length 3 --selection_budget 0.01 --mini_batch_size 1000 --retention_rate 0.3
```
<details>
    <summary>Options</summary>
    0.3, 0.5, and 0.9
</details>

### Lambda 0 value
Adjust the lambda_0 parameter with the '<b>--lambda_0'</b> option to adjust the weight assigned to uniqueness scores during the sample selection process.

Example:
```bash
python Mateen.py  --dataset_name "IDS2017" --window_size 50000 --shift_threshold 0.05 --performance_thres 0.99 --max_ensemble_length 3 --selection_budget 0.01 --mini_batch_size 1000 --retention_rate 0.3 --lambda_0 0.1
```
<details>
    <summary>Options</summary>
    0.1, 0.5, and 1.0
</details>


## Hyperparameter Selection
For further details about the hyperparameter selection, please refer to the main paper, Appendix C.

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
