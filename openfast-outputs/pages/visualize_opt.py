'''This is the page for visualize the optimization results'''

'''
For understanding:
Callback function - Add controls to build the interaction. Automatically run this function whenever changes detected from either Input or State. Update the output.
'''
# TODO: Choose a folder either OpenFAST opt output or RAFT opt output
# TODO: Find alternatives to Global variables
# TODO: Reusable fig creation function?

# Import Packages
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
from utils.utils import *

# Not working, import locally for now..
# from weis.aeroelasticse.FileTools import load_yaml
# from weis.visualization.utils import read_cm, load_OMsql, parse_contents

register_page(
    __name__,
    name='Optimize',
    top_nav=True,
    path='/optimize'
)


def read_log(log_file_path):
    global df       # set the dataframe as a global variable to access it from the get_trace() function.
    log_data = load_OMsql(log_file_path)
    df = parse_contents(log_data)
    # df.to_csv('RAFT/log_opt.csv', index=False)


# We are using card container where we define sublayout with rows and cols.
def layout():
    # log_file_path = 'visualization_demo/log_opt.sql'
    log_file_path = 'RAFT/log_opt.sql'
    read_log(log_file_path)
    
    # Layout for visualizing Conv-trend data
    convergence_layout = dbc.Card(
                            [
                                dbc.CardHeader('Convergence trend data', className='cardHeader'),
                                dbc.CardBody([
                                    dcc.Loading(
                                        html.Div([
                                            html.H5('Select Y-channel:'),
                                            dcc.Dropdown(id='signaly', options=sorted(df.keys()), multi=True),      # Get 'signaly' channels from user. Related function: update_graphs()
                                            dcc.Graph(id='conv-trend', figure=empty_figure()),                      # Initialize with empty figure and update with 'update-graphs() function'. Related function: update_graphs()
                                        ])
                                    )
                                ])
                            ], className='divBorder')

    # Layout for visualizing Specific Iteration data - hidden in default
    iteration_with_dlc_layout = dbc.Collapse(
                                    dbc.Card([
                                            dbc.CardHeader(id='dlc-output-iteration', className='cardHeader'),      # Related function: update_dlc_outputs()
                                            dbc.CardBody([
                                                dcc.Loading(html.Div(id='dlc-iteration-data'))                      # Related function: update_dlc_outputs()
                                            ])], className='divBorder'),
                                    id = 'collapse',
                                    is_open=False)
    

    layout = dbc.Row([
                dbc.Col(convergence_layout, width=6),
                dbc.Col(iteration_with_dlc_layout, width=6),

                # Modal Window layout for visualizing Outlier timeseries data
                dcc.Loading(dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle(html.Div(id='outlier-header'))),                                 # Related function: display_outlier()
                    dbc.ModalBody(html.Div(id='outlier'))],                                                         # Related function: display_outlier()
                    id='outlier-div',
                    size='xl',
                    is_open=False))
    ])

    return layout



###############################################
#   Convergence trend data related functions
###############################################

def get_trace(label):
    '''
    Add the line graph (trace) for each channel (label)
    '''
    assert isinstance(df[label][0], np.ndarray) == True
    trace_list = []
    print(f'{label}:')
    print(df[label])
    print("num of rows: ", len(df[label]))       # The number of rows
    print("first cell: ", df[label][0])          # size of the list in each cell
    print("dimension: ", df[label][0].ndim)
    
    # Need to parse the data depending on the dimension of values
    if df[label][0].ndim == 0:      # For single value
        print('Single value')
        trace_list.append(go.Scatter(y = df[label], mode = 'lines+markers', name = label))
    
    elif df[label][0].ndim == 1:    # For 1d-array
        print('1D-array')
        for i in range(df[label][0].size):
            trace_list.append(go.Scatter(y = df[label].str[i], mode = 'lines+markers', name = label+'_'+str(i)))        # Works perfectly fine with 'visualization_demo/log_opt.sql'

    # TODO: how to viz 2d/3d-array cells?
    elif df[label][0].ndim == 2:    # For 2d-array
        print('2D-array')
        print('we cannot visualize arrays with more than one dimension')

    else:
        print('Need to add function..')
        print('we cannot visualize arrays with more than one dimension')
    

    return trace_list



@callback(Output('conv-trend', 'figure'),
          Input('signaly', 'value'))
def update_graphs(signaly):
    '''
    Draw figures showing convergence trend with selected channels
    '''
    if signaly is None:
        raise PreventUpdate

    # Add subplots for multiple y-channels vertically
    fig = make_subplots(
        rows = len(signaly),
        cols = 1,
        shared_xaxes=True,
        vertical_spacing=0.05)

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



###############################################
# DLC related functions
###############################################

@callback(Output('collapse', 'is_open'),
          Input('conv-trend', 'clickData'),
          State('collapse', 'is_open'))
def toggle_iteration_with_dlc_layout(clickData, is_open):
    '''
    If iteration has been clicked, open the card layout on right side.
    '''
    return toggle(clickData, is_open)


@callback(Output('dlc-output-iteration', 'children'),
          Output('dlc-iteration-data', 'children'),
          Input('conv-trend', 'clickData'))
def update_dlc_outputs(clickData):
    '''
    Once iteration has been clicked from the left convergence graph, analyze:
    1) What # of iteration has been clicked
    2) Corresponding iteration related optimization output files
    '''
    if clickData is None:
        raise PreventUpdate
    
    global iteration, stats, cm
    iteration = clickData['points'][0]['x']
    title_phrase = f'DLC Analysis on Iteration {iteration}'

    # TODO: won't need this once set up with file tree
    if not iteration in [0, 1, 51]:
        return title_phrase, html.Div([html.H5("Please select other iteration..")])
    
    stats = read_per_iteration(iteration)
    case_matrix_path = 'visualization_demo/openfast_runs/rank_0/case_matrix.yaml'
    cm = read_cm(case_matrix_path)
    multi_indices = sorted(stats.reset_index().keys()),

    # Define sublayout that includes user customized panel for visualizing DLC analysis
    sublayout = html.Div([
        html.H5("X Channel Statistics"),
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
            value='min'
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
            value='min'
        )], className='radio-group'),
        html.H5("X Channel"),
        dcc.Dropdown(id='x-channel', options=sorted(set([multi_key[0] for idx, multi_key in enumerate(multi_indices[0])])), value='Wind1VelX'),
        html.H5("Y Channel"),
        dcc.Dropdown(id='y-channel', options=sorted(set([multi_key[0] for idx, multi_key in enumerate(multi_indices[0])])), value=['Wind1VelY', 'Wind1VelZ'], multi=True),
        dcc.Graph(id='dlc-output', figure=empty_figure()),                          # Related functions: update_dlc_plot()
    ])

    return title_phrase, sublayout


@callback(Output('dlc-output', 'figure'),
          Input('x-stat-option', 'value'),
          Input('y-stat-option', 'value'),
          Input('x-channel', 'value'),
          Input('y-channel', 'value'))
def update_dlc_plot(x_chan_option, y_chan_option, x_channel, y_channel):
    '''
    Once required channels and stats options have been selected, draw figures that demonstrate DLC analysis.
    It will show default figure with default settings.
    '''
    if stats is None or x_channel is None or y_channel is None:
        raise PreventUpdate

    fig = plot_dlc(cm, stats, x_chan_option, y_chan_option, x_channel, y_channel)

    return fig

def plot_dlc(cm, stats, x_chan_option, y_chan_option, x_channel, y_channels):
    '''
    Function from:
    https://github.com/WISDEM/WEIS/blob/main/examples/16_postprocessing/rev_DLCs_WEIS.ipynb

    Plot user specified stats option for each DLC over user specified channels
    '''
    dlc_inds = {}

    dlcs = cm[('DLC', 'Label')].unique()
    for dlc in dlcs:
        dlc_inds[dlc] = cm[('DLC', 'Label')] == dlc     # dlcs- key: dlc / value: boolean array
    
    # Add subplots for multiple y-channels vertically
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


###############################################
# Outlier related functions
###############################################

@callback(Output('outlier-div', 'is_open'),
          Input('dlc-output', 'clickData'),
          State('outlier-div', 'is_open'))
def toggle_outlier_timeseries_layout(clickData, is_open):
    '''
    Once user assumes a point as outlier and click that point, open the modal window showing the corresponding time series data.
    '''
    return toggle(clickData, is_open)


@callback(Output('outlier-header', 'children'),
          Output('outlier', 'children'),
          Input('dlc-output', 'clickData'))
def display_outlier(clickData):
    '''
    Once outlier has been clicked, show corresponding optimization run.
    '''
    if clickData is None:
        raise PreventUpdate
    
    print("clickData\n", clickData)
    of_run_num = clickData['points'][0]['pointIndex']
    print("corresponding openfast run: ", of_run_num)

    global timeseries_data
    filename, timeseries_data = get_timeseries_data(of_run_num, stats, iteration)
    print(timeseries_data)

    sublayout = dcc.Loading(html.Div([
        html.H5("Channel to visualize timeseries data"),
        dcc.Dropdown(id='time-signaly', options=sorted(timeseries_data.keys()), value=['Wind1VelX', 'Wind1VelY', 'Wind1VelZ'], multi=True),
        dcc.Graph(id='time-graph', figure=empty_figure())
    ]))

    return filename, sublayout


@callback(Output('time-graph', 'figure'),
          Input('time-signaly', 'value'))
def update_timegraphs(signaly):
    '''
    Function to visualize the time series data graph
    '''
    if signaly is None:
        raise PreventUpdate

    fig = make_subplots(rows = 1, cols = 1)
    for col_idx, label in enumerate(signaly):
        fig.append_trace(go.Scatter(
            x = timeseries_data['Time'],
            y = timeseries_data[label],
            mode = 'lines',
            name = label),
            row = 1,
            col = 1)
    
    return fig


