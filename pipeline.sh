#!/bin/bash

# Path to your virtual environment
VENV_PATH="./.venv"

# Check if python is available
if ! command -v python &> /dev/null
then
    echo "Python not found, attempting to activate virtual environment..."
    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
        echo "Virtual environment activated."
    else
        echo "Virtual environment not found at $VENV_PATH."
        exit 1
    fi
else
    echo "Python is available."
fi


python --version

echo "Introduction to Machine Learning - Project 1: k-IBL Pipeline"

# Pipeline steps:
# 1. Data Preprocessing
echo "Starting Data Preprocessing..."
echo "NOT IMPLEMENTED YET"

# 2. Model Training
echo "Starting Model Training..."
echo "Running experiments for the two main datasets..."
python experiments.py --datasets credit-a pen-based --similarity_metrics all --voting_schemes all --retention all

echo "Running experiments for the two extra datasets..."
python experiments.py --datasets vowel grid --similarity_metrics all --voting_schemes all --retention all



# 3. Statistical Tests for Model Selection
echo "Starting statistical tests for model selection..."

python stats.py --alpha 0.1 --datasets credit-a pen-based --test_name k-IBL_hyperparameters_2_datasets
python stats.py --alpha 0.1 --datasets all --test_name k-IBL_hyperparameters_all_datasets


# 4. Weighting Techniques
echo "Starting weighting techniques experiments..."
echo "Running weighting techniques experiments for the two main datasets..."
python experiments.py --datasets credit-a pen-based --similarity_metrics euc --voting_schemes bc --retention ar --weighting all --ks 5

echo "Running weighting techniques experiments for the two extra datasets..."
python experiments.py --datasets vowel grid --similarity_metrics euc --voting_schemes bc --retention ar --weighting all --ks 5

# 5. Statistical Tests for Weighting Techniques
echo "Statistical tests for weighting techniques..."
python stats.py --alpha 0.1 --datasets credit-a pen-based --test_name k-IBL_weighting_2_datasets --base_dir "res/weighted"
python stats.py --alpha 0.05 --datasets credit-a pen-based --test_name k-IBL_weighting_2_datasets --base_dir "res/weighted"
python stats.py --alpha 0.01 --datasets credit-a pen-based --test_name k-IBL_weighting_2_datasets --base_dir "res/weighted"
python stats.py --alpha 0.1 --datasets all --test_name k-IBL_weighting_all_datasets --base_dir "res/weighted"
python stats.py --alpha 0.05 --datasets all --test_name k-IBL_weighting_all_datasets --base_dir "res/weighted"
python stats.py --alpha 0.01 --datasets all --test_name k-IBL_weighting_all_datasets --base_dir "res/weighted"


# 6. Reduction Techniques
echo "Starting reduction techniques experiments..."

echo "Running reduction techniques experiments for the two main datasets..."
python experiments.py --datasets credit-a pen-based --similarity_metrics euc --voting_schemes bc --retention ar --reduction_technique all --ks 5

echo "Running reduction techniques experiments for the two extra datasets..."
python experiments.py --datasets vowel grid --similarity_metrics euc --voting_schemes bc --retention ar --reduction_technique all --ks 5


# 7. Statistical Tests for Reduction Techniques
echo "Starting statistical tests for reduction techniques..."
python stats.py --alpha 0.1 --datasets credit-a pen-based --test_name k-IBL_reduction_2_datasets --base_dir "res/reduction"
python stats.py --alpha 0.05 --datasets credit-a pen-based --test_name k-IBL_reduction_2_datasets --base_dir "res/reduction"
python stats.py --alpha 0.01 --datasets credit-a pen-based --test_name k-IBL_reduction_2_datasets --base_dir "res/reduction"
python stats.py --alpha 0.1 --datasets all --test_name k-IBL_reduction_all_datasets --base_dir "res/reduction"
python stats.py --alpha 0.05 --datasets all --test_name k-IBL_reduction_all_datasets --base_dir "res/reduction"
python stats.py --alpha 0.01 --datasets all --test_name k-IBL_reduction_all_datasets --base_dir "res/reduction"

# 6. SVM
echo "Starting SVM experiments..."
echo "Running SVM experiments for the two main datasets..."
python SVM.py --datasets credit-a pen-based -s

echo "Running SVM experiments for the two extra datasets..."
python SVM.py --datasets vowel grid -s

# 7. Statistical Tests for SVM
echo "Starting statistical tests for svm..."
python stats.py --alpha 0.1 --datasets credit-a pen-based --test_name SVM_2_datasets --base_dir "res/svm"
python stats.py --alpha 0.1 --datasets all --test_name SVM_all_datasets --base_dir "res/svm"


# 8. SVM with reduction techniques
# Print in red color that in order to carry out this step, lines 246-248 in SVM.py must be commented out. The same for line 251.
echo -e "\e[31mIMPORTANT: To carry out SVM with reduction techniques experiments, please comment out lines 246-248 and line 251 in SVM.py\e[0m"
# Only allow to continue with this part of the pipeline if the user confirms
read -p "Do you want to continue? (y/n): " choice
if [[ "$choice" != "y" ]]; then
    echo "Skipping SVM with reduction techniques experiments."
    exit 0
else
    echo "Continuing with SVM with reduction techniques experiments."
    # Say in color that we assume that the user has commented out the lines in SVM.py, otherwise, the results will be done for all SVM configurations, not for the best one.
    echo -e "\e[32mAssuming that you have commented out the necessary lines in SVM.py...\e[0m"
    echo "Starting SVM with reduction techniques experiments..."
    echo "Running SVM with reduction techniques experiments for the two main datasets..."
    python SVM.py --datasets credit-a pen-based --reduction_technique all -s --output_dir "res/svm_reduction"

    echo "Running SVM with reduction techniques experiments for the two extra datasets..."
    python SVM.py --datasets vowel grid --reduction_technique all -s --output_dir "res/svm_reduction"


    # 9. Statistical Tests for SVM with reduction techniques
    echo "Starting statistical tests for svm with reduction techniques..."
    python stats.py --alpha 0.1 --datasets credit-a pen-based --test_name SVM_reduction_2_datasets --base_dir "res/svm_reduction"
    python stats.py --alpha 0.01 --datasets credit-a pen-based --test_name SVM_reduction_2_datasets --base_dir "res/svm_reduction"
    python stats.py --alpha 0.05 --datasets credit-a pen-based --test_name SVM_reduction_2_datasets --base_dir "res/svm_reduction"
    python stats.py --alpha 0.1 --datasets all --test_name SVM_reduction_all_datasets --base_dir "res/svm_reduction"
    python stats.py --alpha 0.01 --datasets all --test_name SVM_reduction_all_datasets --base_dir "res/svm_reduction"
    python stats.py --alpha 0.05 --datasets all --test_name SVM_reduction_all_datasets --base_dir "res/svm_reduction"

fi

# Visualizations for the reduction techniques on the different datasets and the complete datasets

echo "Starting complete dataset visualizations..."
python dataset.py
echo "Starting dimensionality reduction visualizations..."
python reduction.py --datasets credit-a pen-based vowel grid --mcnn --pca
python reduction.py --datasets credit-a pen-based vowel grid --mcnn --tsne
python reduction.py --datasets credit-a pen-based vowel grid --icf -k 3 --pca
python reduction.py --datasets credit-a pen-based vowel grid --icf -k 5 --pca
python reduction.py --datasets credit-a pen-based vowel grid --icf -k 7 --pca
python reduction.py --datasets credit-a pen-based vowel grid --icf -k 3 --tsne
python reduction.py --datasets credit-a pen-based vowel grid --icf -k 5 --tsne
python reduction.py --datasets credit-a pen-based vowel grid --icf -k 7 --tsne
python reduction.py --datasets credit-a pen-based vowel grid --allknn -k 3 --pca
python reduction.py --datasets credit-a pen-based vowel grid --allknn -k 5 --pca
python reduction.py --datasets credit-a pen-based vowel grid --allknn -k 7 --pca
python reduction.py --datasets credit-a pen-based vowel grid --allknn -k 3 --tsne
python reduction.py --datasets credit-a pen-based vowel grid --allknn -k 5 --tsne
python reduction.py --datasets credit-a pen-based vowel grid --allknn -k 7 --tsne