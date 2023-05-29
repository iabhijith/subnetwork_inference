# Installation requirements
- Install the conda enviroment locally with `conda env create -f environment.yml`  
- To train the models on a gpu cluster (lisa/snellius) follow the below instructions
    - Directly clone the repo to the cluster with `git clone git@github.com:iabhijith/subnetwork_inference.git`
    - Alternatively upload the repo to the cluster with `rsync -av subnetwork_inference __username__@snellius.surf.nl:~/`
    - Install the environemnt on the cluster using `sbatch scripts/create_env.job`

# Running the code
## Running the UCI experiments
- It is suggested to train the models on a gpu cluster (Instructions apply only for SLURM based clusters)
- To train the models on a gpu cluster follow the below instructions
    - There are 3 different job files for each of the datasets in the `scripts` folder
        - `scripts/train_wine.job`
        - `scripts/train_kin8nm.job`
        - `scripts/train_protein.job`
    - To run the job file on the cluster use for eg. `sbatch scripts/train_wine.job`
    - The job is configured to train models for 5(3) different seeds for both GAP and STANDARD splits
    - You can change the following hyperparameters as required
        - seed: the seeds for which the experiments are run
        - trainer.la.subset_size: the sizes of the subnetworks for inference
        - trainer.epochs: the number of epochs for which the models are trained
        - trainer.patience: the early stopping patience
    - Other hyperparameters can be changed if required either in the job file or in the `configuration/uci.yaml` file directly
    - First the MAP models are trained followed by subnetwork inference using laplace approximations.
    - The results are stored in the `results` folder
    - The models are stored in the `checkpoints` folder

- The code can also be run locally (suggested to run on a gpu)
    - To run the code locally use `python main.py`
    - The hyperparameters can be configured using the `configuration/uci.yaml` file 
    - Alternatively, the hyperparameters be configured using the command line
        - For eg. `python main.py trainer.epochs=100 trainer.la.subset_size=100`
        
- Once the models are trained and results for each individual model are written to the `results` folder,
    - Copy the `results` folders to the local machine to visualize the results
    - The results can be visualized in the `subnetwork_inference.ipynb` notebook
    - The notebook can only be run locally


## Running the Snelson1D experiments
- The experiments can be reproduced by running the `snelson1d.ipynb` notebook
- The notebook can only be run locally. Please ensure that the environment is installed locally

## Running the additional experiments
- The experiments can be reproduced by running the `subnetwork_analysis.ipynb` notebook
- The notebook can only be run locally. Please ensure that the environment is installed locally
   