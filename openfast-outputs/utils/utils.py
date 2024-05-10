'''
Various functions for help visualizing WEIS outputs
'''
# from weis.aeroelasticse.FileTools import load_yaml        # TODO: Add
import pandas as pd
import numpy as np
import openmdao.api as om
import plotly.graph_objects as go
import os

try:
    import ruamel_yaml as ry
except Exception:
    try:
        import ruamel.yaml as ry
    except Exception:
        raise ImportError('No module named ruamel.yaml or ruamel_yaml')

def read_cm(cm_file):
    """
    Function originally from:
    https://github.com/WISDEM/WEIS/blob/main/examples/16_postprocessing/rev_DLCs_WEIS.ipynb

    Parameters
    __________
    cm_file : The file path for case matrix

    Returns
    _______
    cm : The dataframe of case matrix
    dlc_inds : The indices dictionary indicating where corresponding dlc is used for each run
    """
    cm_dict = load_yaml(cm_file, package=1)
    cnames = []
    for c in list(cm_dict.keys()):
        if isinstance(c, ry.comments.CommentedKeySeq):
            cnames.append(tuple(c))
        else:
            cnames.append(c)
    cm = pd.DataFrame(cm_dict, columns=cnames)
    
    return cm

def parse_contents(data):
    """
    Function from:
    https://github.com/WISDEM/WEIS/blob/main/examples/09_design_of_experiments/postprocess_results.py
    """
    collected_data = {}
    for key in data.keys():
        if key not in collected_data.keys():
            collected_data[key] = []
        
        for key_idx, _ in enumerate(data[key]):
            if isinstance(data[key][key_idx], int):
                collected_data[key].append(np.array(data[key][key_idx]))
            elif len(data[key][key_idx]) == 1:
                try:
                    collected_data[key].append(np.array(data[key][key_idx][0]))
                except:
                    collected_data[key].append(np.array(data[key][key_idx]))
            else:
                collected_data[key].append(np.array(data[key][key_idx]))
    
    df = pd.DataFrame.from_dict(collected_data)

    return df


def load_OMsql(log):
    """
    Function from :
    https://github.com/WISDEM/WEIS/blob/main/examples/09_design_of_experiments/postprocess_results.py
    """
    # logging.info("loading ", log)
    cr = om.CaseReader(log)
    rec_data = {}
    cases = cr.get_cases('driver')
    for case in cases:
        for key in case.outputs.keys():
            if key not in rec_data:
                rec_data[key] = []
            rec_data[key].append(case[key])
    
    return rec_data


# TODO: Remove
def load_yaml(fname_input, package=0):
    """
    Function from:
    https://github.com/WISDEM/WEIS/blob/main/weis/aeroelasticse/FileTools.py
    """
    if package == 0:
        with open(fname_input) as f:
            data = yaml.safe_load(f)
        return data

    elif package == 1:
        with open(fname_input, 'r') as myfile:
            text_input = myfile.read()
        myfile.close()
        ryaml = ry.YAML()
        return dict(ryaml.load(text_input))


# TODO: Add below functions under WEIS/weis/visualization/utils.py
def read_per_iteration(iteration):
    iteration_path = 'visualization_demo/openfast_runs/rank_0/iteration_{}'.format(iteration)
    stats = pd.read_pickle(iteration_path+'/summary_stats.p')
    # dels = pd.read_pickle(iteration_path+'/DELs.p')
    # fst_vt = pd.read_pickle(iteration_path+'/fst_vt.p')

    return stats

# TODO: Add
def get_timeseries_data(run_num, stats, iteration):
    
    stats = stats.reset_index()     # make 'index' column that has elements of 'IEA_22_Semi_00, ...'
    print("stats\n", stats)
    filename = stats.loc[run_num, 'index'].to_string()      # filenames are not same - stats: IEA_22_Semi_83 / timeseries/: IEA_22_Semi_0_83.p
    if filename.split('_')[-1].startswith('0'):
        filename = ('_'.join(filename.split('_')[:-1])+'_0_'+filename.split('_')[-1][1:]+'.p').strip()
    else:
        filename = ('_'.join(filename.split('_')[:-1])+'_0_'+filename.split('_')[-1]+'.p').strip()
    
    # visualization_demo/openfast_runs/rank_0/iteration_0/timeseries/IEA_22_Semi_0_0.p
    timeseries_path = 'visualization_demo/openfast_runs/rank_0/iteration_{}/timeseries/{}'.format(iteration, filename)
    print('timeseries_path:\n', timeseries_path)
    timeseries_data = pd.read_pickle(timeseries_path)

    return filename, timeseries_data


# TODO: Add
def empty_figure():
    '''
    Draw empty figure showing nothing once initialized
    '''
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig

# TODO: Add
def toggle(click, is_open):
    if click:
        return not is_open
    return is_open


# TODO: Add
def store_dataframes(var_files):
    dfs = []
    for idx, file_path in var_files.items():
        df = pd.read_csv(file_path, skiprows=[0,1,2,3,4,5,7], delim_whitespace=True)
        # dfs[idx] = df
        dfs.append({idx: df.to_dict('records')})
    
    return dfs


# TODO: Add
def get_file_info(file_path):
    file_name = file_path.split('/')[-1]
    file_abs_path = os.path.abspath(file_path)
    file_size = round(os.path.getsize(file_path) / (1024**2), 2)
    creation_time = os.path.getctime(file_path)
    modification_time = os.path.getmtime(file_path)

    file_info = {
        'file_name': file_name,
        'file_abs_path': file_abs_path,
        'file_size': file_size,
        'creation_time': creation_time,
        'modification_time': modification_time
    }

    return file_info