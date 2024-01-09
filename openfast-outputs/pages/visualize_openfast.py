'''This is the page for visualizing table and plots'''

# Import Packages
from dash import Dash, Input, Output, State, callback, dcc, html, register_page
from dash.exceptions import PreventUpdate
import copy
import plotly.express as px
import plotly.graph_objects as go
import plotly.tools as tls
from plotly.subplots import make_subplots
import dash_mantine_components as dmc
import pandas as pd
import logging


register_page(
    __name__,
    name='OpenFAST',
    top_nav=True,
    path='/open_fast'
)

def layout():
    layout = html.Div([
        html.H1(
            [
                "OpenFAST Visualization"
            ]
        ),
        html.Div(id='openfast-div')
    ])
    return layout


@callback(
        Output('openfast-div', 'children'),
        Input('store', 'data')
)
def analyze(store):
    if store == {}:        # Nothing happens if data is not uploaded yet..
        logging.warning(f"Upload data first..")
        raise PreventUpdate

    else:
        logging.warning(f"Analyze function..")
        df = pd.DataFrame(store)
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
                dcc.Graph(id="line")
            ]
        )

# Add controls to build the interaction
@callback(
    Output(component_id="line", component_property="figure"),
    Input(component_id="signalx", component_property="value"),
    [Input(component_id="signaly", component_property="value")],
    Input(component_id="plotOption", component_property="value"),
    Input('store', 'data')
)
def draw_graphs(signalx, signaly, plotOption, store):

    # fig = tls.make_subplots(rows=1, cols=1, shared_xaxes=True, verical_spacing=0.009, horizontal_spacing=0.009)
    # fig = go.Figure()

    if (signalx is None) or (signaly == []):        # Do not update the graph
        raise PreventUpdate

    else:
        df = pd.DataFrame(store)
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
