'''This is the page for visualizing table and plots of OpenFAST output'''

'''
For understanding:
Callback function - Add controls to build the interaction. Automatically run this function whenever changes detected from either Input or State. Update the output.
'''

# TODO: Need to solve following warning error - A nonexistent object was used in an `Input` of a Dash callback. The id of this object is `signalx` and the property is `value`.
#       This is caused by the fact that 'signalx' defined in sublayout.


# Import Packages
import dash_bootstrap_components as dbc
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
from utils.utils import *

register_page(
    __name__,
    name='OpenFAST',
    top_nav=True,
    path='/open_fast'
)

# We are using card container where we define sublayout with rows and cols.
def layout():
    file_upload_layout = dcc.Upload(
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
                        )
    
    layout = dbc.Row([
                # Data to share over functions
                dcc.Store(id='store', data={}),
                # Starts with Pop-up window
                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle('File Upload')),
                    dbc.ModalBody(file_upload_layout)],
                    id='upload-div',
                    size='xl',
                    is_open=True
                ),
                # Left column is for description layout
                dbc.Col(dcc.Loading(html.Div(id='output-data-upload')), width=4),        # related function: show_data_contents(), analyze()
                # Right column is for graph layout
                dbc.Col(dcc.Loading(html.Div(id='graph-div')), width=8)                  # related function: draw_graphs()
            ])
    
    return layout


@callback(Output('upload-div', 'is_open'),
          Input('upload-data', 'contents'),
          State('upload-div', 'is_open'))
def toggle_modal(n1, is_open):
    '''
    Once we get the file selected from the user (upload-data), close the pop-up window (upload-div)
    '''
    return toggle(n1, is_open)



@callback(Output('store', 'data'),
          Input('upload-data', 'contents'))
def get_data(contents):
    '''
    Once we get the file selected from the user (upload-data), parse the data and saved it as 'store' id.
    '''
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), skiprows=[0,1,2,3,4,5,7], delim_whitespace=True)

        return df.to_dict('records')


@callback(Output('output-data-upload', 'children'),
          Input('store', 'data'),
          State('upload-data', 'filename'),
          State('upload-data', 'last_modified'))
def show_data_contents(store, name, date):
    '''
    Once we parse the data from get_deta(), add the sublayout to the main layout where the div is defined as (output-data-upload).
    Hence, show the filename and table for the description layout.
    '''
    df = pd.DataFrame(store)

    if name is not None:
        table_layout = dbc.Card(
                        [
                            dbc.CardHeader(f'File name: {name}', className='cardHeader'),
                            dcc.Loading(dbc.CardBody([
                                    html.H5(f'Date: {datetime.datetime.fromtimestamp(date)}'),
                                    dash_table.DataTable(
                                        data=store,
                                        columns=[{'name': i, 'id': i} for i in df.columns],
                                        fixed_columns = {'headers': True, 'data': 1},
                                        page_size=10,
                                        style_table={'height': '300px', 'overflowX': 'auto', 'overflowY': 'auto'}),
                                    html.Div(id='openfast-div')
                            ]))
                        ], className='divBorder')
        
        return table_layout
    

@callback(
        Output('openfast-div', 'children'),
        Input('store', 'data')
)
def analyze(store):
    '''
    Once we parse the data from get_deta(), add the sublayout to the main layout where the div is defined as (output-data-upload).
    Hence, add channels/dropdown lists for the description layout.
    '''
    if store == {}:        # Prevent this function to be called when data is not uploaded yet
        logging.warning(f"Upload data first..")
        raise PreventUpdate

    else:
        logging.info(f"Analyze function..")
        df = pd.DataFrame(store)
        return html.Div(
            [
                html.H5("Signal-y"),
                dcc.Dropdown(
                    id='signaly', options=sorted(df.keys()), value=['Wind1VelX', 'Wind1VelY', 'Wind1VelZ'], multi=True),          # options look like ['Azimuth', 'B1N1Alpha', ...]. select ['Wind1VelX', 'Wind1VelY', 'Wind1VelZ'] as default value
                html.H5("Signal-x"),
                dcc.Dropdown(
                    id='signalx', options=sorted(df.keys()), value='Time'),                                                       # select 'Time' as default value
                html.Br(),
                html.H5("Plot Options"),
                dcc.RadioItems(
                    id='plotOption', options=['single plot', 'multiple plot'], value='single plot', inline=True)                  # Select 'single plot' as default value
            ]
        )

@callback(
    Output('graph-div', 'children'),
    Input('signalx', 'value'),
    [Input('signaly', 'value')],
    Input('plotOption', 'value'),
    Input('store', 'data')
)
def draw_graphs(signalx, signaly, plotOption, store):
    '''
    Whenever signalx, signaly, plotOption has been entered, draw the graph.
    Create figure with that setting and add that figure to the graph layout.
    Note that we set default settings (see analyze() function), it will show corresponding default graph.
    You can dynamically change signalx, signaly, plotOption, and it will automatically update the graph.
    '''

    # If something missing, don't call this function (= don't draw the figure)
    if store is None or signalx is None or signaly is None or plotOption is None:
        raise PreventUpdate
    
    else:
        df = pd.DataFrame(store)

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
        graph_layout = dbc.Card(
                            [
                                dbc.CardHeader("Graphs", className='cardHeader'),
                                dbc.CardBody([
                                    dcc.Graph(figure=fig)
                                ])
                            ], className='divBorder')
        
        return graph_layout
        

