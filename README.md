# WEIS-Visualization
Full-stack development for WEIS input/output visualization

# How to Start
```
conda env create -f environment.yml                                   # Create conda environment from yml
conda activate dash                                                   # Activate the new environment
git clone https://github.com/sora-ryu/WEIS-Visualization.git          # Clone the repository
cd openfast-outputs/
python mainApp.py                                                     # Run the App
```

# Documentation
## Outputs from WEIS and each location
Once optimization is done, all logs over all iterations are recorded in log_opt.sql file. Each iteration folder contains time series data from each openfast runs and the summary of statistics. summary_stats.p file is multi-index dataframe where it summarizes the statistics for each run. The statistics metrics include 'min, max, std, mean, median, labs and integrated'. The information on what openfast file indices corresponds to can be located in case_matrix.txt and case_matrix.yaml files.

Below is the sample file structure from WEIS Ouputs.

```bash
WEIS Output/
├── IEA-22-280-RWT-analysis.yaml
├── IEA-22-280-RWT-modeling.yaml
├── IEA-22-280-RWT.csv
├── IEA-22-280-RWT.mat
├── IEA-22-280-RWT.npz
├── IEA-22-280-RWT.pkl
├── IEA-22-280-RWT.xlsx
├── IEA-22-280-RWT.yaml
├── log_opt.sql                                  # The optimization iteration logs generated from openmdao
└── openfast_runs/
    └── rank_0/
        ├── case_matrix.txt                      # Where we dump all cases for the iterations
        ├── case_matrix.yaml                     # yaml version of case_matrix.txt
        ├── iteration_0/
        │   ├── DELs.p
        │   ├── fst_vt.p                         # Related to input ?
        │   ├── summary_stats.p                  # The summary of optimization
        │   └── timeseries/
        │       ├── IEA_22_Semi_0_0.p
        │       ├── IEA_22_Semi_0_1.p
        │       │         .
        │       │         .
        │       │         .
        │       └── IEA_22_Semi_0_n.p            # Where (n+1) openfast runs has been processed
        ├── iteration_1/
        │        .
        │        .
        │        .
        └── iteration_m/                         # Where we have m iterations in this specific weis optimization example
```

# To-Do Checklist
## Quarter 1
- [x] Build multi-page app prototype
- [x] Implement an OpenFAST output visualization page

## Quarter 2
- [x] Implement Optimization page - optimization convergence data
- [ ] Implement Optimization page - DLC/OpenFAST statistics
- [ ] Implement Optimization page - outlier DLC with OpenFAST time-series