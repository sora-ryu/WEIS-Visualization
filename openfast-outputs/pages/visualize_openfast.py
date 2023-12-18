'''This is the page for visualizing table and plots'''

# Import Packages
import dash
from dash import Dash, Input, Output, callback, dcc, html, dash_table
import copy
import plotly.express as px
import plotly.graph_objects as go
import plotly.tools as tls
from plotly.subplots import make_subplots
import dash_mantine_components as dmc
import pandas as pd


layout = html.Div(id='body-div')


@callback(
        Output('body-div', 'children'),
        Input('stored-data', 'data')
)
def get_data(data):
    df = pd.DataFrame(data)
    return html.Div(
        [
            html.H5("Signal-y"),
            dcc.Dropdown(
                id="signaly", options=sorted(df.keys()), multi=True
            ),
            html.H5("Signal-x"),
            dcc.Dropdown(
                id="signalx", options=sorted(df.keys())
            ),
            html.Br(),
            html.H5("Plot Options"),
            dcc.RadioItems(
                options=['single plot', 'multiple plot'], value='single plot', inline=True, id='plotOption'
            ),
            html.Br(),
            dcc.Graph(id="line"),
        ]
    )

# Add controls to build the interaction
@callback(
    Output(component_id="line", component_property="figure"),
    Input(component_id="signalx", component_property="value"),
    [Input(component_id="signaly", component_property="value")],
    Input(component_id="plotOption", component_property="value"),
    Input('stored-data', 'data')
)
def draw_graphs(signalx, signaly, plotOption, data):

    # fig = tls.make_subplots(rows=1, cols=1, shared_xaxes=True, verical_spacing=0.009, horizontal_spacing=0.009)
    # fig = go.Figure()

    df = pd.DataFrame(data)

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

    
    return fig
