# PSA-MolGen: An Innovative Multi-Scale Generative Model for Structure-Based De Novo Drug Design

![pipeline](images/mod2.png)

### Requirements

Model training is written in `pytorch==1.11.0` and uses `keras==2.4.0` for data loaders. `RDKit==2020.09.5` is needed for molecule manipulation.


### Creat a new environment in conda 


## Pre-training

### Data Preparation
For the training a npy file is needed. We used subset of the Zinc dataset, using only the drug-like.The same clear target specific datasets were obtained from DUD-E database (http://dude.docking.org/targets).

In the `data/zinc` folder there will be the `zinc.smi` file that is required for the preparing data step.

`python data_prepare.py     --input ./data/zinc/zinc.csv 
                            --output ./data/zinc/zinc.npy`
                            --mode 1
python generate_features.py -input input_PDB_testing.dat -out output_features.csv

## Training Model

python train.py -i ./data/zinc/zinc1.npy  -fn_train ./data/v2020/output_features.csv -fn_test ./data/v2020/output_features.csv


## Generation

python generation.py 


## Transfer Learning 

The process of transfer learning is the same as it is in zinc data sets, using train_trs.py files when training models.

python train_trs.py -i ./data/zinc/cdk2.npy  -fn_train ./data/cdk2_output_features.csv -fn_validate ./data/cdk2_output_features.csv 

 
