'''This is the page for visualize the WISDEM outputs specialized in blade properties'''

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
import os

from utils.utils import *
from wisdem.glue_code.runWISDEM import load_wisdem

register_page(
    __name__,
    name='WISDEM',
    top_nav=True,
    path='/wisdem_blade'
)


def layout():
    # Read pickle file
    # refturb, _, _ = load_wisdem("/Users/sryu/Desktop/FY24/WEIS/WISDEM/examples/02_reference_turbines/outputs/refturb_output.pkl")   # openmdao object
    # tree = refturb.model.list_outputs()
    # print('refturb\n', tree)

    # Read csv file
    # global refturb_variables
    # filepath = "/Users/sryu/Desktop/FY24/WEIS/WISDEM/examples/02_reference_turbines/outputs/refturb_output.csv"
    # refturb = pd.read_csv(filepath)
    # refturb_variables = refturb.set_index('variables').to_dict('index')
    # # print("df variables\n", df_variables)   # e.g., ['tower_grid.length': {'units': 'm', 'values': '[87.7]', 'description': 'Scalar of the tower length computed along its curved axis. A standard straight tower will be as high as long.'}, ...]

    # Read numpy file
    global refturb, refturb_variables
    npz_filepath = "/Users/sryu/Desktop/FY24/WEIS/WISDEM/examples/02_reference_turbines/outputs/refturb_output.npz"
    csv_filepath = "/Users/sryu/Desktop/FY24/WEIS/WISDEM/examples/02_reference_turbines/outputs/refturb_output.csv"     # For description
    refturb = np.load(npz_filepath)
    refturb_variables = pd.read_csv(csv_filepath).set_index('variables').to_dict('index')

    # Blade property channels
    x = 'rotorse.rc.s'
    ys = ['rotorse.rc.chord_m', 'rotorse.re.pitch_axis', 'rotorse.theta_deg']       # channel name at npz
    ys_struct_log = ['rotorse.EA_N', 'rotorse.EIxx_N*m**2', 'rotorse.EIyy_N*m**2', 'rotorse.GJ_N*m**2']
    ys_struct = ['rotorse.rhoA_kg/m']
    
    layout = html.Div([
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    html.B("Blade property channels with description"),
                    dcc.Loading(html.P(children=get_description([x]+ys+ys_struct+ys_struct_log)))
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(dcc.Graph(figure=draw_blade_shape(x, ys, ys_struct, ys_struct_log)))
                    ]),
                    dbc.Col([
                        dcc.Loading(dcc.Graph(figure=draw_blade_structure(x, ys, ys_struct, ys_struct_log)))
                    ])
                ])
            ])
        )
    ])
    return layout


def get_description(channel_list):
    des_list = []
    # Where channel names are saved differently..
    npz_to_csv = {'rotorse.rc.chord_m': 'rotorse.rc.chord', 'rotorse.theta_deg': 'rotorse.theta', 'rotorse.EA_N': 'rotorse.EA', 'rotorse.EIxx_N*m**2': 'rotorse.EIxx', 'rotorse.EIyy_N*m**2': 'rotorse.EIyy', 'rotorse.GJ_N*m**2': 'rotorse.GJ', 'rotorse.rhoA_kg/m': 'rotorse.rhoA'}
    for chan in channel_list:
        if chan in npz_to_csv.keys():
            value = refturb_variables[npz_to_csv[chan]]
            des = npz_to_csv[chan]
        else:
            value = refturb_variables[chan]
            des = chan
        
        if not pd.isna(value['units']):
            des += ' ('+value['units']+'): '+value['description']
        else:
            des += ' : '+value['description']

        des_list.append(html.P(des))

    return des_list



def draw_blade_shape(x, ys, ys_struct, ys_struct_log):
    
    fig = make_subplots(rows = 2, cols = 1, shared_xaxes=True)

    for y in ys:
        if y == 'rotorse.theta_deg':
            fig.append_trace(go.Scatter(
                x = refturb[x],
                y = refturb[y],
                mode = 'lines+markers',
                name = y),
                row = 2,
                col = 1)
        else:
            fig.append_trace(go.Scatter(
                x = refturb[x],
                y = refturb[y],
                mode = 'lines+markers',
                name = y),
                row = 1,
                col = 1)
        
    
    fig.update_xaxes(title_text=f'rotorse.rc.s', row=2, col=1)
    fig.update_layout(title='Blade properties')
    
    return fig


def draw_blade_structure(x, ys, ys_struct, ys_struct_log):

    fig = make_subplots(specs=[[{"secondary_y": True}], [{"secondary_y": False}]], rows=2, cols=1, shared_xaxes=True)
    for y in ys_struct:
        fig.add_trace(go.Scatter(
            x = refturb[x],
            y = refturb[y],
            mode = 'lines+markers',
            name = y),
            row = 2,
            col = 1)
    for y in ys_struct_log:
        fig.add_trace(go.Scatter(
            x = refturb[x],
            y = refturb[y],
            mode = 'lines+markers',
            name = y),
            secondary_y=True,
            row = 1,
            col = 1)
    
    fig.update_yaxes(type="log", secondary_y=True)
    fig.update_xaxes(title_text=f'rotorse.rc.s', row=2, col=1)
    fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
    fig.update_yaxes(title_text="<b>secondary</b> yaxis title with log", secondary_y=True)
    fig.update_layout(title='Blade Structure Properties')

    return fig
