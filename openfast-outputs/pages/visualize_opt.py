'''This is the page for visualize the optimization results'''

import dash_bootstrap_components as dbc
from dash import html, register_page, callback, Input, Output, dcc, State
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
import json

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


def empty_figure():
    '''
    Draw empty figure showing nothing once initialized
    '''
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


def layout():
    log_file_path = 'visualization_demo/log_opt.sql'
    read_log(log_file_path)

    layout = html.Div([

        # Headline
        # html.H1(['Optimization']),

        # Visualize Conv-trend data
        html.Div([
            html.H5('Y-Channel to visualize from Convergence trend data'),
            dcc.Dropdown(id='signaly', options=sorted(df.keys()), multi=True),
            dcc.Graph(id='conv-trend', figure=empty_figure()),
        ], style = {'width': '49%', 'display': 'inline-block', 'margin-left': '15px'}),

        # Visualize Specific Iteration data
        html.Div(dcc.Loading(html.Div([
            html.H3(id='dlc-output-iteration', style={'textAlign':'center'}),
            html.Div(id='dlc-iteration-data'),
        ])), style = {'width': '49%', 'display': 'inline-block', 'margin-left': '15px'}),

        # Visualize Outlier timeseries data with modal
        dcc.Loading(dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle('Header')),
            dbc.ModalBody(html.Div(id='outlier'))],
            id='outlier-div',
            size='lg',
            is_open=False
        ))
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
          Output('dlc-iteration-data', 'children'),
          Input('conv-trend', 'clickData'))
def update_dlc_outputs(clickData):

    if clickData is None:
        raise PreventUpdate
    
    global iteration
    iteration = clickData['points'][0]['x']
    title_phrase = f'DLC Analysis on Iteration {iteration}'
    if not iteration in [0, 1, 51]:
        return title_phrase, None
    
    global stats
    global cm
    stats = read_per_iteration(iteration)
    case_matrix_path = 'visualization_demo/openfast_runs/rank_0/case_matrix.yaml'
    cm = read_cm(case_matrix_path)
    # fig = plot_dlc(cm, stats)

    multi_indices = sorted(stats.reset_index().keys()),

    sublayout = html.Div([
        html.H5("X Channel Statistics"),
        # dcc.RadioItems(options=['min', 'max', 'std', 'mean', 'median', 'abs', 'integrated'], value='mean', inline=True, id='x-stat-option'),
        html.Div([dbc.RadioItems(
            id='x-stat-option',
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {'label': 'min', 'value': 'min'},
                {'label': 'max', 'value': 'max'},
                {'label': 'std', 'value': 'std'},
                {'label':  'mean', 'value': 'mean'},
                {'label': 'median', 'value': 'median'},
                {'label': 'abs', 'value': 'abs'},
                {'label': 'integrated', 'value': 'integrated'}],
        )], className='radio-group'),
        html.H5("Y Channel Statistics"),
        html.Div([dbc.RadioItems(
            id='y-stat-option',
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {'label': 'min', 'value': 'min'},
                {'label': 'max', 'value': 'max'},
                {'label': 'std', 'value': 'std'},
                {'label':  'mean', 'value': 'mean'},
                {'label': 'median', 'value': 'median'},
                {'label': 'abs', 'value': 'abs'},
                {'label': 'integrated', 'value': 'integrated'}],
        )], className='radio-group'),
        html.H5("X Channel"),
        dcc.Dropdown(id='x-channel', options=sorted(set([multi_key[0] for idx, multi_key in enumerate(multi_indices[0])]))),
        html.H5("Y Channel"),
        dcc.Dropdown(id='y-channel', options=sorted(set([multi_key[0] for idx, multi_key in enumerate(multi_indices[0])])), multi=True),
        dcc.Graph(id='dlc-output', figure=empty_figure()),
    ])

    '''
    # Radio Items Option
    sublayout = html.Div([
        html.H5("X Channel Statistics"),
        dcc.RadioItems(options=['min', 'max', 'std', 'mean', 'median', 'abs', 'integrated'], value='mean', inline=True, id='x-stat-option'),
        html.H5("Y Channel Statistics"),
        dcc.RadioItems(options=['min', 'max', 'std', 'mean', 'median', 'abs', 'integrated'], value='max', inline=True, id='y-stat-option'),
        html.H5("X Channel"),
        dcc.Dropdown(id='x-channel', options=sorted(set([multi_key[0] for idx, multi_key in enumerate(multi_indices[0])]))),
        html.H5("Y Channel"),
        dcc.Dropdown(id='y-channel', options=sorted(set([multi_key[0] for idx, multi_key in enumerate(multi_indices[0])])), multi=True),
        dcc.Graph(id='dlc-output', figure=empty_figure()),
    ])
    '''

    return title_phrase, sublayout


def read_per_iteration(iteration):
    iteration_path = 'visualization_demo/openfast_runs/rank_0/iteration_{}'.format(iteration)
    stats = pd.read_pickle(iteration_path+'/summary_stats.p')
    dels = pd.read_pickle(iteration_path+'/DELs.p')
    fst_vt = pd.read_pickle(iteration_path+'/fst_vt.p')

    return stats


@callback(Output('dlc-output', 'figure'),
          Input('x-stat-option', 'value'),
          Input('y-stat-option', 'value'),
          Input('x-channel', 'value'),
          Input('y-channel', 'value'))
def update_dlc_plot(x_chan_option, y_chan_option, x_channel, y_channel):
    if stats is None or x_channel is None or y_channel is None:
        raise PreventUpdate

    fig = plot_dlc(cm, stats, x_chan_option, y_chan_option, x_channel, y_channel)

    return fig

def plot_dlc(cm, stats, x_chan_option, y_chan_option, x_channel, y_channels):
    '''
    Function from:
    https://github.com/WISDEM/WEIS/blob/main/examples/16_postprocessing/rev_DLCs_WEIS.ipynb

    Plot channel maxima vs mean wind speed for each DLC - channel: user specify
    '''
    dlc_inds = {}

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

@callback(Output('outlier-div', 'is_open'),
          Input('dlc-output', 'clickData'),
          State('outlier-div', 'is_open'))
def toggle_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open

@callback(Output('outlier', 'children'),
          Input('dlc-output', 'clickData'))
def display_outlier(clickData):

    if clickData is None:
        raise PreventUpdate
    
    print("clickData\n", clickData)
    of_run_num = clickData['points'][0]['pointIndex']
    print("corresponding openfast run: ", of_run_num)

    timeseries_data = get_timeseries_data(of_run_num, stats, iteration)
    print(timeseries_data)

    return json.dumps(clickData, indent=2)


def get_timeseries_data(run_num, stats, iteration):
    print("stats\n", stats)
    stats = stats.reset_index()     # make 'index' column that has elements of 'IEA_22_Semi_00, ...'
    print("stats\n", stats)
    filename = stats.loc[run_num, 'index'].to_string()      # filenames are not same - stats: IEA_22_Semi_83 / timeseries/: IEA_22_Semi_0_83.p
    if filename.split('_')[-1].startswith('0'):
        filename = ('_'.join(filename.split('_')[:-1])+'_0_'+filename.split('_')[-1][1:]+'.p').strip()
    else:
        filename = ('_'.join(filename.split('_')[:-1])+'_0_'+filename.split('_')[-1]+'.p').strip()
    print("filename: ", filename)
    # visualization_demo/openfast_runs/rank_0/iteration_0/timeseries/IEA_22_Semi_0_0.p
    timeseries_path = 'visualization_demo/openfast_runs/rank_0/iteration_{}/timeseries/{}'.format(iteration, filename)
    print('timeseries_path\n', timeseries_path)
    timeseries_data = pd.read_pickle(timeseries_path)

    return timeseries_data


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
