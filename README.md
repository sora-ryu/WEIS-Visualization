# WEIS-Visualization
Full-stack development for WEIS input/output visualization

# User Guide - How to Start
Make sure that 'test.yaml' file located under openfast-outpus/ folder.
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

## Guideline for building customized viz function for WISDEM output
1. Add NavItem from mainApp.py. Add dropdownmenu item at line 33 and declare href link to the page.
2. Create page under pages/ folder. Try to set the name with visualize_wisdem_<>.py.
3. Register the page with href link you defined from Step 1.

# To-Do Checklist
## Quarter 1
- [x] Build multi-page app prototype
- [x] Implement an OpenFAST output visualization page

## Quarter 2
- [x] Implement Optimization page - optimization convergence data
- [x] Implement Optimization page - DLC/OpenFAST statistics
- [x] Implement Optimization page - outlier DLC with OpenFAST time-series with modal window
- [x] Implement Optimization page - update layout to handle a bunch of plots
- [x] Implement WISDEM Viz page - blade

## Quarter 3
- [x] Read user preferences
- [x] Update the user preferences, variable settings from yaml file
- [x] Trivial function updates - accept only yaml file for input, solve warning errors (nonexistent object at the callback functions)
- [x] Maintain the progress even if changing tabs
- [x] Find file paths from dir tree from yaml file
- [x] Implement WISDEM Viz page - cost
- [x] Merge into WEIS - verify if it works well with 'weis-env'

<!-- ## ETC (If needed)
- [ ] Merge into WEIS - version match doesn't work.. assuming the reason=python version. Change the code to gain proper plot..
- [ ] Implement WISDEM Viz page - general
- [ ] Implement WISDEM Viz page - doc for customized viz function
- [ ] Improve UI - drag and drop card -->