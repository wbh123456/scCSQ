# Single-cell Deep Hashing (scDeepHash)
A deep-learning-based single-cell classification model with central similarity hashing method.

## Quick Start:
Install dependencies
`pip3 install -r requirements.txt`
Train scDeepHash on Baron Human
`python3 --dataset BaronHuman`

### Options
  `--l_r`             learning rate
  `--lamb`           lambda of quantization loss
  `--lr_decay`   learning rate decay
  `--n_layers`   number of layers
  `--epochs`       number of epochs to run
  `--dataset {TM,BaronHuman,Zheng68K,AMB,XIN,CellBench,X10v2,Pancreatic,AlignedPancreatic}`
                        dataset to train against
                        
  *For more options, please see https://github.com/wbh123456/scDeepHash/blob/f687c3edbd09352b44295d3a51abcec2f2c92efb/util.py*                      
## Built-in datasets
##### Intra-dataset:
 - Baron Human
 - TM
 - Zheng68K
 - AMB
 - XIN

##### Inter-dataset:
 - CellBench
 - X10v2
 - Pancreatic
 - AlignedPancreatic
 
## Establish a venv
- `python3 -m venv .venv`
- `pip3 install -r requirements.txt`