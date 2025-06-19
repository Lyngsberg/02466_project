# 02466_project
Project for DTU course 02466
```text
02466_project/
├── .gitignore
├── LICENSE
├── sweep_deep_NN_final_model_training.yaml
├── sweep_deep_PN_final_model_training.yaml
├── sweep_deep.yaml
├── sweep.yaml
├── README.md
├── fagproject/
│   ├── .gitignore
│   ├── .pre-commit-config.yaml
│   ├── LICENSE
│   ├── pyproject.toml
│   ├── requirements_dev.txt
│   ├── requirements.txt
│   ├── tasks.py
│   ├── .github/
│   ├── configs/
│   ├── data/
│   ├── docs/
│   │   ├── mkdocs.yml
│   │   └── source/
│   │       └── index.md
│   ├── figs/
│   ├── models/
│   ├── notebooks/
│   ├── reports/
│   │   └── figures/
│   └── src/
│       └── project/
│           ├── train.py
│           ├── data.py
│           ├── plot.py
│           ├── barplot.py
│           ├── conf_max.py
│           ├── performance_statistics.py
│           ├── sweep_train.py
│           ├── Deep_PN/
│           │   ├── NN_deep.py
│           │   ├── PN_model_triang_deep.py
│           │   ├── sweep_deep.py
│           │   └── train_deep.py
│           ├── Triang_vs_nonTriang/
│           │   ├── plot.py
│           │   ├── PN_model_for_triang.py
│           │   └──  train_triangulizePN_vs_nontrianglizePN.py
│           ├── Wandb_train.py
│           ├── visualize.py
│           ├── PN_models.py
│           └── NN_models.py
└── wandb/


```
```bash
git clone https://github.com/Lyngsberg/02466_project.git
cd 02466_project
pip install -r fagproject/requirements.txt
```

```text
to run the scripts for the deep models:
```
```bash
python fagproject/src/project/Deep_PN/train_deep.py
```
