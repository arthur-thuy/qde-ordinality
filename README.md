# Ordinality in Discrete-level Question Difficulty Estimation

This repository contains the code to reproduce the results of the paper _Ordinality in Discrete-level Question Difficulty Estimation: Introducing Balanced DRPS and OrderedLogitNN_.

The paper is available here as an arXiv preprint.

```
cd "/home/abthuy/Documents/PhD research/qde-ordinal/src"
conda activate qde_ordinal
```


## Workflow

Download the RACE ([here](https://huggingface.co/datasets/ehovy/race)), RACE-c ([here](https://huggingface.co/datasets/tasksource/race-c)), and ARC ([here](https://allenai.org/data/arc)) datasets. Together, RACE and RACE-c form the RACE++ dataset. Save the datasets in the `data/raw/` folder.

Prepare the datasets by running:
```
python data_preparation.py
```

Create the environment from the `environment.yml` file and activate it:
```
conda env create -f environment.yml
conda activate qde_ordinal
```

The configuration files are located in the folder `src/config/`, e.g., `arc_bert_bal_rps/`. Run the experiments with:
```
python main.py arc_bert_bal_rps
```

Finally, inspect the results by running the `analysis.ipynb` notebook.


<!-- ## Cite as

If you use this code in your workflow or scientific publication, please cite the corresponding paper:
```
TODO: add bibtex entry
``` -->

