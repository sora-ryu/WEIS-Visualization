'Main Page where we get the input file'

# Import Packages
import dash
from dash import Dash, Input, Output, State, callback, dcc, html, dash_table
import io
import base64
import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.tools as tls
from plotly.subplots import make_subplots
import dash_mantine_components as dmc
import pandas as pd

# Connect to your app pages
from pages import visualize_openfast


# Initialize the app - Internally starts the Flask Server
# Incorporate a Dash Mantine theme
external_stylesheets = [dmc.theme.DEFAULT_COLORS]
app = Dash(__name__, external_stylesheets = external_stylesheets)

app.layout = html.Div([
    # dcc.Graph(id='MyGraph', animate=True),
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='stored-data', storage_type='session'),
    # Upload data
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
    html.Div(id='output-data-upload')
])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), skiprows=[0,1,2,3,4,5,7], delim_whitespace=True)
    except Exception as e:
        print(e)
        return html.Div([
            'There is some error on processing this file..'
        ])
    
    return html.Div([
        html.H5(f'File name: {filename}'),
        html.H6(f'Date: {datetime.datetime.fromtimestamp(date)}'),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=10
        ),
        dcc.Store(id='stored-data', data=df.to_dict('records')),

        html.Hr(),      # Horizontal line

        html.Button('Visualize with plots', id='submit', n_clicks=0),       # refresh: controls whether or not the page will refresh when the link is clicked
        html.Div(id='page-content', children=[])
        # For debugging, display the raw contents provided by the web browser
        # html.Div('Raw Content'),
        # html.Pre(contents[0:200] + '...', style={
        #     'whiteSpace': 'pre-wrap',
        #     'wordBreak': 'break-all'
        # })

    ])

@callback(Output('output-data-upload', 'children'),
          Input('upload-data', 'contents'),
          State('upload-data', 'filename'),
          State('upload-data', 'last_modified'))
def show_data_contents(contents, names, dates):
    if contents is not None:
        # 1) Single File version
        children = parse_contents(contents, names, dates)

        # 2) Multiple files version
        # children = [
        #     parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
        # ]

        return children

@callback(Output('page-content', 'children'),
          Input('submit', 'n_clicks'))
def display_page(n_clicks):
    if n_clicks > 0:
        return visualize_openfast.layout

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",port="3030")