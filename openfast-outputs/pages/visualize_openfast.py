'''This is the page for visualizing table and plots of OpenFAST output'''

# Import Packages
from dash import Dash, Input, Output, State, callback, dcc, html, dash_table, register_page
from dash.exceptions import PreventUpdate
import copy
import base64
import io
import datetime
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
        dcc.Upload(
            id='upload-data', children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # multiple=True         # Allow multiple files to be uploaded
        ),
        html.Div(id='output-data-upload'),
        html.Div(id='openfast-div')
    ])
    return layout

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), skiprows=[0,1,2,3,4,5,7], delim_whitespace=True)
        data=df.to_dict('records')
    except Exception as e:
        print(e)
        return html.Div([
            'There is some error on processing this file..'
        ]), {}
    
    return html.Div([
        html.H5(f'File name: {filename}'),
        html.H6(f'Date: {datetime.datetime.fromtimestamp(date)}'),

        dash_table.DataTable(
            data=data,
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=10
        )
        # html.Div(id='page-content', children=[])
        # For debugging, display the raw contents provided by the web browser
        # html.Div('Raw Content'),
        # html.Pre(contents[0:200] + '...', style={
        #     'whiteSpace': 'pre-wrap',
        #     'wordBreak': 'break-all'
        # })

    ]), data


@callback(Output('store', 'data'),
          Input('upload-data', 'contents'))
def get_data(contents):
    if contents is not None:
        # Store data in a dcc.Store in app.py
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), skiprows=[0,1,2,3,4,5,7], delim_whitespace=True)
        return df.to_dict('records')



@callback(Output('output-data-upload', 'children'),
          Input('store', 'data'),
          State('upload-data', 'filename'),
          State('upload-data', 'last_modified'))
def show_data_contents(store, name, date):

    df = pd.DataFrame(store)

    if name is not None:
        return html.Div([
            html.H5(f'File name: {name}'),
            html.H6(f'Date: {datetime.datetime.fromtimestamp(date)}'),

            dash_table.DataTable(
                data=store,
                columns=[{'name': i, 'id': i} for i in df.columns],
                page_size=10
            )
        ])
    

@callback(
        Output('openfast-div', 'children'),
        Input('store', 'data')
)
def analyze(store):
    if store == {}:        # Nothing happens if data is not uploaded yet..
        logging.warning(f"Upload data first..")
        raise PreventUpdate

    else:
        logging.info(f"Analyze function..")
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
