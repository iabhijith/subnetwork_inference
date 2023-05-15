# Installation requirements
- Install the enviroment.yml file with conda
- Activate the enviroment with `conda activate subnetwork_inference`

# Running the code
- Run the code with `python main.py`
- Change configurations in `configuration/uci.yaml`
    - subset_of_weights: {"all", "last_layer", "subnetwork"}
    - hessian_structure: {"diagonal", "full", "kron}