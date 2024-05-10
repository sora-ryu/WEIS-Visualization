'''This is the page for visualizing table and plots of OpenFAST output'''

'''
For understanding:
Callback function - Add controls to build the interaction. Automatically run this function whenever changes detected from either Input or State. Update the output.
'''

# TODO: Save changed variable settings into input yaml file again

# Import Packages
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, MATCH, ALL, callback, dcc, html, dash_table, register_page
from dash.exceptions import PreventUpdate
import copy
import base64
import io
import os
import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.tools as tls
from plotly.subplots import make_subplots
import dash_mantine_components as dmc
import pandas as pd
import logging
from utils.utils import *

register_page(
    __name__,
    name='OpenFAST',
    top_nav=True,
    path='/open_fast'
)

file_indices = ['file1', 'file2']

@callback(Output('var-openfast', 'data'),
          [[Output(f'df-{idx}', 'data') for idx in file_indices]],
          Output('var-graph', 'data'),
          Input('input-dict', 'data'))
def read_variables(input_dict):
    '''
    Input: 
    '''
    # TODO: Redirect to the home page when missing input yaml file
    if input_dict is None or input_dict == {}:
        raise PreventUpdate
    
    var_openfast = input_dict['userPreferences']['openfast']        # {'file_path': {'file1': 'of-output/NREL5MW_OC3_spar_0.out', 'file2': 'of-output/IEA15_0.out'}, 'graph': {'xaxis': 'Time', 'yaxis': ['Wind1VelX', 'Wind1VelY', 'Wind1VelZ']}}
    var_files = var_openfast['file_path']
    dfs = store_dataframes(var_files)       # [{file1: df1, file2: df2, ... }]
    
    num_files = list(var_files.keys())
    print('num of files: ', num_files)

    var_graphs = var_openfast['graph']

    return var_openfast, dfs, var_graphs


# We are using card container where we define sublayout with rows and cols.
def layout():
    layout = dcc.Loading(html.Div([
                # OpenFAST related Data fetched from input-dict
                dcc.Store(id='var-openfast', data={}),
                # dcc.Store(id='file-indices', data=[]),
                # Dataframe to share over functions - openfast .out file
                html.Div(
                    [dcc.Store(id=f'df-{idx}', data={}) for idx in file_indices]      # dcc.Store(id='df-file1', data={}),          # {file1, df1}
                ),
                # Graph configuration
                dcc.Store(id='var-graph', data={}),
                
                dbc.Card([
                    dbc.CardBody([
                        dbc.InputGroup(
                            [
                                # Layout for showing graph configuration setting
                                html.Div(id='graph-cfg-div', className='text-center'),
                                dbc.Button('Save', id='save-cfg', n_clicks=0)
                            ]
                        )
                    ])
                ]),
                # Append cards per file
                dbc.Row([], id='output')
            ]))
    
    return layout


@callback(Output('graph-cfg-div', 'children'),
          Input('df-file1', 'data'),
          Input('var-graph', 'data'))
def define_graph_cfg_layout(df1, var_graph):
    
    print('var_graph: ', var_graph)
    signalx = var_graph['xaxis']
    signaly = var_graph['yaxis']
    channels = sorted(df1['file1'][0].keys())
    # print(df_dict['file1'][0])          # First row channels

    return html.Div([
                html.Div([
                    html.Label(['Signal-y:'], style={'font-weight':'bold', 'text-align':'center'}),
                    dcc.Dropdown(id='signaly', options=channels, value=signaly, multi=True),          # options look like ['Azimuth', 'B1N1Alpha', ...]. select ['Wind1VelX', 'Wind1VelY', 'Wind1VelZ'] as default value
                ], style = {'float':'left'}),
                html.Div([
                    html.Label(['Signal-x:'], style={'font-weight':'bold', 'text-align':'center'}),
                    dcc.Dropdown(id='signalx', options=channels, value=signalx),          # options look like ['Azimuth', 'B1N1Alpha', ...]. select ['Wind1VelX', 'Wind1VelY', 'Wind1VelZ'] as default value
                ], style = {'float':'left'}),
                html.Div([
                    html.Label(['Plot options:'], style={'font-weight':'bold', 'text-align':'center'}),
                    dcc.RadioItems(id='plotOption', options=['single plot', 'multiple plot'], value='single plot', inline=True),
                ], style = {'float':'left'})
            ])



def define_des_layout(file_info, df):
    file_abs_path = file_info['file_abs_path']
    file_size = file_info['file_size']
    creation_time = file_info['creation_time']
    modification_time = file_info['modification_time']
    
    return html.Div([
                    # File Info
                    html.H5(f'File Path: {file_abs_path}'),
                    html.H5(f'File Size: {file_size} MB'),
                    html.H5(f'Creation Date: {datetime.datetime.fromtimestamp(creation_time)}'),
                    html.H5(f'Modification Date: {datetime.datetime.fromtimestamp(modification_time)}'),
                    html.Br(),

                    # Data Table
                    # dash_table.DataTable(
                    #     data=df,
                    #     columns=[{'name': i, 'id': i} for i in pd.DataFrame(df).columns],
                    #     fixed_columns = {'headers': True, 'data': 1},
                    #     page_size=10,
                    #     style_table={'height': '300px', 'overflowX': 'auto', 'overflowY': 'auto'})
            ])


def update_figure(signalx, signaly, plotOption, df_dict):
    print('here')
    df, = df_dict.values()
    return draw_graph(signalx, signaly, plotOption, pd.DataFrame(df))


for idx in file_indices:
    callback(Output(f'graph-div-{idx}', 'figure'),
                Input('signalx', 'value'),
                Input('signaly', 'value'),
                Input('plotOption', 'value'),
                Input(f'df-{idx}', 'data'))(update_figure)


def draw_graph(signalx, signaly, plotOption, df):
    # Whenever signalx, signaly, plotOption has been entered, draw the graph.
    # Create figure with that setting and add that figure to the graph layout.
    # Note that we set default settings (see analyze() function), it will show corresponding default graph.
    # You can dynamically change signalx, signaly, plotOption, and it will automatically update the graph.

    print(signalx, signaly)
    # Put all traces in one single plot
    if plotOption == 'single plot':
        fig = make_subplots(rows = 1, cols = 1)
        for col_idx, label in enumerate(signaly):
            fig.append_trace(go.Scatter(
                x = df[signalx],
                y = df[label],
                mode = 'lines',
                name = label),
                row = 1,
                col = 1)

    # Put each traces in each separated horizontally aligned subplots
    elif plotOption == 'multiple plot':
        fig = make_subplots(rows = 1, cols = len(signaly))

        for col_idx, label in enumerate(signaly):
            fig.append_trace(go.Scatter(
                x = df[signalx],
                y = df[label],
                mode = 'lines',
                name = label),
                row = 1,
                col = col_idx + 1)
    
    # Define the graph layout where it includes the rendered figure
    return fig



def make_card(idx, file_path, df):
    file_info = get_file_info(file_path)
    file_name = file_info['file_name']
    print('idx: ', idx)

    return dbc.Card([
        dbc.CardHeader(f'File name: {file_name}', className='cardHeader'),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Loading(define_des_layout(file_info, df)), width=4),
                dbc.Col(dcc.Loading(dcc.Graph(id=f'graph-div-{idx}')), width=8)
            ])
        ])
    ])


@callback(Output('output', 'children'),
          Input('var-openfast', 'data'),
          [[Input(f'df-{idx}', 'data') for idx in file_indices]])
def manage_cards(var_openfast, df_dict_list):
    # df_dict_list = [{file1: df1}, {file2: df2}, ...]
    children = []
    for idx, file_path in var_openfast['file_path'].items():            # idx = file1, file2, ...
        df_idx = [d.get(idx, None) for d in df_dict_list][0]
        children.append(make_card(idx, file_path, df_idx))      # Pass: file1, file1.out, df1
    
    return children
