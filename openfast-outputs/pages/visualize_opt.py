'''This is the page for visualize the optimization results'''

from dash import html, register_page, callback, Input, Output, dcc, dash_table
import pandas as pd
import numpy as np
import logging
import yaml
import ruamel_yaml as ry
import openmdao.api as om
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

register_page(
    __name__,
    name='Optimize',
    top_nav=True,
    path='/optimize'
)

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


def read_cm(cm_file):
    """
    Function from:
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


def read_log(log_file_path):
    global df
    log_data = load_OMsql(log_file_path)
    df = parse_contents(log_data)
    # df.to_csv('visualization_demo/log_opt.csv', index=False)


def layout():
    log_file_path = 'visualization_demo/log_opt.sql'
    read_log(log_file_path)

    layout = html.Div([

        # Headline
        html.H1(["Optimization"]),

        # Visualize Conv-trend data
        html.Div([
            html.H5("Signal-y"),
            dcc.Dropdown(id="signaly", options=sorted(df.keys()), multi=True),
            dcc.Graph(id='conv-trend'),
        ], style = {'width': '59%', 'display': 'inline-block'}),

        # Visualize Specific Iteration data
        html.Div([
            html.H3(id='dlc-output-iteration'),
            dcc.Graph(id='dlc-output'),
        ], style = {'width': '39%', 'float': 'right', 'display': 'inline', 'padding': '0 20'})
    ])

    return layout


def get_trace(label):
    
    # The channels that have list values
    list_values = ['floatingse.constr_draft_heel_margin', 'floatingse.constr_fixed_margin', 'floatingse.constr_freeboard_heel_margin']
    trace_list = []
    if label in list_values:
        for i in range(len(df[label][0])):
            trace_list.append(go.Scatter(y = df[label].str[i], mode = 'lines+markers', name = label+'_'+str(i)))
    
    else:
        trace_list.append(go.Scatter(y = df[label], mode = 'lines+markers', name = label))

    return trace_list



@callback(Output('conv-trend', 'figure'),
          Input('signaly', 'value'))
def update_graphs(signaly):

    if signaly is None:
        raise PreventUpdate

    fig = make_subplots(
        rows = len(signaly),
        cols = 1,
        shared_xaxes=True,
        vertical_spacing=0.05)

    # Add traces
    for row_idx, label in enumerate(signaly):
        trace_list = get_trace(label)
        for trace in trace_list:
            fig.add_trace(trace, row=row_idx+1, col=1)
        fig.update_yaxes(title_text=label, row=row_idx+1, col=1)
    
    fig.update_layout(
        height=250 * len(signaly),
        hovermode='x unified',
        title='Convergence Trend from Optimization',
        title_x=0.5)

    fig.update_traces(xaxis='x'+str(len(signaly)))   # Spike line hover extended to all subplots

    fig.update_xaxes(
        spikemode='across+marker',
        spikesnap='cursor',
        title_text='Iteration')

    return fig


@callback(Output('dlc-output-iteration', 'children'),
          Output('dlc-output', 'figure'),
          Input('conv-trend', 'hoverData'))
def update_dlc_outputs(hoverData):

    if hoverData is None:
        raise PreventUpdate
    
    iteration = hoverData['points'][0]['x']
    title_phrase = f'DLC Analysis on Iteration {iteration}'
    if iteration != 0:
        return title_phrase, go.Figure()
    
    stats = read_per_iteration(iteration)
    case_matrix_path = 'visualization_demo/openfast_runs/rank_0/case_matrix.yaml'
    cm = read_cm(case_matrix_path)
    fig = plot_dlc(cm, stats)

    return title_phrase, fig


def read_per_iteration(iteration):
    
    iteration_path = 'visualization_demo/openfast_runs/rank_0/iteration_{}'.format(iteration)
    stats = pd.read_pickle(iteration_path+'/summary_stats.p')
    dels = pd.read_pickle(iteration_path+'/DELs.p')
    fst_vt = pd.read_pickle(iteration_path+'/fst_vt.p')

    return stats


def plot_dlc(cm, stats):
    '''
    Function from:
    https://github.com/WISDEM/WEIS/blob/main/examples/16_postprocessing/rev_DLCs_WEIS.ipynb

    Plot channel maxima vs mean wind speed for each DLC - channel: user specify
    '''
    dlc_inds = {}
    y_channels = ['GenSpeed', 'PtfmPitch', 'PtfmRoll', 'PtfmYaw']
    x_channel = 'Wind1VelX'
    y_chan_option = 'max'
    x_chan_option = 'mean'

    dlcs = cm[('DLC', 'Label')].unique()
    for dlc in dlcs:
        dlc_inds[dlc] = cm[('DLC', 'Label')] == dlc     # dlcs- key: dlc / value: boolean array
    

    fig = make_subplots(
        rows = len(y_channels),
        cols = 1,
        vertical_spacing=0.1)

    # Add traces
    for row_idx, y_channel in enumerate(y_channels):
        for dlc, boolean_dlc in dlc_inds.items():
            x = stats.reset_index()[x_channel][x_chan_option].to_numpy()[boolean_dlc]
            y = stats.reset_index()[y_channel][y_chan_option].to_numpy()[boolean_dlc]
            trace = go.Scatter(x=x, y=y, mode='markers', name='dlc_'+str(dlc))
            fig.add_trace(trace, row=row_idx+1, col=1)
        fig.update_yaxes(title_text=f'{y_chan_option.capitalize()} {y_channel}', row=row_idx+1, col=1)
        fig.update_xaxes(title_text=f'{x_chan_option.capitalize()} {x_channel}', row=row_idx+1, col=1)

    fig.update_layout(
        height=300 * len(y_channels))
    
    return fig


# Don't need
def visualize_stats(stats):
    features = ['Wind1VelX', 'Wind1VelY', 'Wind1VelZ']
    fig = make_subplots(rows = 1, cols = 1)
    for feature in features:
        feature_min = stats[feature]['min']
        feature_max = stats[feature]['max']
        feature_std = stats[feature]['std']
        feature_mean = stats[feature]['mean']   # (n,1) where n-runs has been implemented for optimization
        feature_median = stats[feature]['median']
        # logging.info(feature_mean)
        # logging.info(feature_mean[2])       # Works (start index from 0)
        fig.add_trace(go.Scatter(
            y=[feature_mean[2]],
            name = feature + '_mean')
        )

    # fig, axes = plt.subplots(nrows=len(features), sharex=True)

    return fig
