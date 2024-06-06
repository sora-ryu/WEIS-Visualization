from dash import html, register_page
from dash import dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from utils.utils import *
import yaml
import base64
import io

register_page(
    __name__,
    name='Home',
    top_nav=True,
    path='/'
)

def layout():

    file_upload_layout = dcc.Upload(
                            id='input-data', children=html.Div([
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
                            multiple=False         # Do not allow multiple files to be uploaded
                        )
    layout = dbc.Row([
                # Confirm Dialog for input file
                dcc.ConfirmDialog(
                    id='confirm-input',
                    message='Please select yaml file to continue..'
                ),
                # Starts with Pop-up window
                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle('Configuration Input File Upload')),
                    dbc.ModalBody(file_upload_layout)],
                    id='input-div',
                    size='xl',
                    is_open=True
                ),
                dbc.Col(dcc.Loading(html.Div(id='input-cfg-div')), width=4),
    ])

    return layout


@callback(Output('input-div', 'is_open'),
          Input('input-dict', 'data'),
          State('input-div', 'is_open'))
def toggle_modal(n1, is_open):
    '''
    Once we get the input configuration file from user, close the pop-up window (upload-div)
    '''
    return toggle(n1, is_open)


def parse_yaml(contents):
    '''
    Parse the data contents in dictionary format
    '''
    content_type, content_string = contents.split(',')      # content_type:  data:application/x-yaml;base64
    decoded = base64.b64decode(content_string)
    dict = yaml.safe_load(decoded)
    # print("input file dict:\n", dict)
    
    return dict


@callback(Output('confirm-input', 'displayed'),
          Output('input-dict', 'data'),
          Output('input-cfg-div', 'children'),
          Input('input-data', 'contents'),
          State('input-data', 'filename'),
          prevent_initial_call=True)
def check_input_file(contents, filename):
    '''
    Store data in mainApp.py so that it's accessible over pages.
    Show if input file data has been loaded and parsed successfully
    '''
    if contents is None:
        raise PreventUpdate
    
    if 'yaml' in filename:
        print('\nInput Filename: ', filename)
        input_dict = parse_yaml(contents)

        return False, input_dict, html.Div([html.H5("Uploaded successfully")])          # TODO: Show file tree instead?
    
    return True, None, None