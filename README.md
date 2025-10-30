# IML_Labs_2025_MAI

## Environment Setup

The python version used for this project is 3.11.13. To create a virtual environment and install the required packages, run the following commands:

```bash
python3.11 -m venv <path_to_venv>
source <path_to_venv>/bin/activate  # On Windows use `<path_to_venv>\Scripts\activate`
pip install -r requirements.txt
```

## File Structure

- `stats.py`: Contains code and classes to perform statistical analysis and visualize results. It can be run from the command line with various options. `stats.py -h`:

    ```code
    usage: Friedman and Nemenyi Tests [-h] [--alpha ALPHA] [--base_dir BASE_DIR] [--datasets DATASETS [DATASETS ...]] --test_name TEST_NAME [--output_path OUTPUT_PATH]

    Perform Friedman and Nemenyi tests on model results.

    options:
    -h, --help            show this help message and exit
    --alpha ALPHA         Significance level for the tests.
    --base_dir BASE_DIR   Base directory where the results files are stored.
    --datasets DATASETS [DATASETS ...]
                            Datasets to include in the analysis. Use 'all' to include all datasets in the base directory.
    --test_name TEST_NAME
                            Name of the test being run
    --output_path OUTPUT_PATH
                            Path to save the output results
    ```
    The results are stored in the specified output path, including text files and visualizations of the significance matrix. The default path where results are stored is `results/stat_test/<test_name>/`.

- `k_ibl.py`: This is one of the core files of the project. It implements the `KIBLearner` class, which encapsulates the k-Instance Based Learning algorithm with feature weighting, voting schemes and retention policies. It also supports different distance metrics and instance reduction techniques.

- `experiments.py`: This script is responsible for running the experiments on the selected datasets with various hyperparameter configurations. It allows users to specify datasets, similarity metrics, voting schemes, and retention policies through command-line arguments. The results of the experiments are saved in JSON format for further analysis. `experiments.py -h`:
  ```code
  usage: experiments.py [-h] [--dataset_dir DATASET_DIR] [--datasets DATASETS [DATASETS ...]] [--ks KS [KS ...]] [--retention {nr,ar,dc,dd,all} [{nr,ar,dc,dd,all} ...]]
                      [--similarity_metrics {euc,cos,ivdm,heom,gwhsm,all} [{euc,cos,ivdm,heom,gwhsm,all} ...]] [--voting_schemes {mp,bc,all} [{mp,bc,all} ...]] [--output_dir OUTPUT_DIR]

    Run K-IBL experiments on specified datasets with various hyperparameter combinations.

    options:
    -h, --help            show this help message and exit
    --dataset_dir DATASET_DIR
                            Directory where preprocessed datasets are stored.
    --datasets DATASETS [DATASETS ...]
                            List of dataset names to run experiments on. Use 'all' to run on all available datasets.
    --ks KS [KS ...]      List of k values for the K-IBL algorithm.
    --retention {nr,ar,dc,dd,all} [{nr,ar,dc,dd,all} ...]
                            List of retention policies to use. Use 'all' to include all available policies.
    --similarity_metrics {euc,cos,ivdm,heom,gwhsm,all} [{euc,cos,ivdm,heom,gwhsm,all} ...]
                            List of similarity metrics to use. Use 'all' to include all available metrics.
    --voting_schemes {mp,bc,all} [{mp,bc,all} ...]
                            List of voting schemes to use. Use 'all' to include all available schemes.
    --output_dir OUTPUT_DIR
                            Directory where output files will be saved.
    ```

- `pipeline.sh`: A shell script to automate the execution of the different parts of the project, including data preprocessing, model training, evaluation, and statistical analysis. It calls the relevant Python scripts with appropriate arguments based on the analysis that we have carried out to obtain the results shown in the report. It can be executed from the command using:
    - `bash pipeline.sh`
    - `./pipeline.sh` (after giving execute permissions with `chmod u+x pipeline.sh`)

- `requirements.txt`: Lists the Python packages required for the project.
- `utils.py`: Contains utility functions used across different scripts in the project.