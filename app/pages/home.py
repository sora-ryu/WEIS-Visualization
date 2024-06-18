from dash import html, register_page
from dash import dcc, Input, Output, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

register_page(
    __name__,
    name='Home',
    top_nav=True,
    path='/'
)

def layout():

    layout = dbc.Row([
                dbc.Col(dcc.Loading(html.Div(id='input-cfg-div')), width=4),
    ])

    return layout


@callback(Output('input-cfg-div', 'children'),
          Input('input-dict', 'data'))
def check_input_file(contents):
    '''
    Store data in mainApp.py so that it's accessible over pages.
    Show if input file data has been loaded and parsed successfully
    '''
    if contents is None:
        raise PreventUpdate
    
    if contents == {}:
        return html.Div([html.H5("Empty content..")])
    
    return html.Div([html.H5("Uploaded successfully")])          # TODO: Show file tree instead?
    