from dash import Dash, Input, Output, State, callback, dcc, html, dash_table, register_page
from dash.exceptions import PreventUpdate
import base64
import io
import pandas as pd
import datetime
import logging

register_page(
    __name__,
    name='Data Upload',
    top_nav=True,
    path='/data_upload'
)

def layout():
    layout = html.Div([
        # Upload and store data
        # dcc.Store(id='stored-data'),
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
