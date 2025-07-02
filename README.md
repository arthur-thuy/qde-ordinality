# Ordinality in Discrete-level Question Difficulty Estimation

This repository contains the code to reproduce the results of the paper _Ordinality in Discrete-level Question Difficulty Estimation: Introducing Balanced DRPS and OrderedLogitNN_.

The paper is available [here](https://arxiv.org/abs/2507.00736) as an arXiv preprint.

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


## Cite as

If you use this code in your workflow or scientific publication, please cite the corresponding paper:
```
@article{thuy2025ordinality,
  title={Ordinality in Discrete-level Question Difficulty Estimation: Introducing Balanced DRPS and OrderedLogitNN},
  author={Thuy, Arthur and Loginova, Ekaterina and Benoit, Dries F},
  journal={arXiv preprint arXiv:2507.00736},
  year={2025},
  doi={10.48550/arXiv.2507.00736}
}
```
