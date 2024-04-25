'''Main Page where we get the input file'''

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
import dash_bootstrap_components as dbc
import pandas as pd
import logging


# Initialize the app - Internally starts the Flask Server
# Incorporate a Dash Mantine theme
external_stylesheets = [dbc.themes.BOOTSTRAP]
APP_TITLE = "WEIS Visualization APP"
app = Dash(__name__, external_stylesheets = external_stylesheets, suppress_callback_exceptions=True, title=APP_TITLE, use_pages=True)

# Build Navigation Bar
# Each pages are registered on each python script under the pages directory.
navbar = dbc.NavbarSimple(
    children = [
        dbc.NavItem(dbc.NavLink("Home", href='/')),
        dbc.NavItem(dbc.NavLink("OpenFAST", href='/open_fast')),
        dbc.NavItem(dbc.NavLink("Optimize", href='/optimize')),
        dbc.DropdownMenu(
            [dbc.DropdownMenuItem('Blade', href='/wisdem_blade'), dbc.DropdownMenuItem('Cost', href='/wisdem_cost'), dbc.DropdownMenuItem('General', href='/wisdem_general')],
            label="WISDEM",
            nav=True
        ),
        dbc.NavItem(dbc.NavLink("3D Visualization", href='/3d_vis'))
    ],
    brand = APP_TITLE,
    color = "dark",
    dark = True,
    className = "menu-bar"
)

# Wrap app with loading component
# Whenever it needs some time for loading data, small progress bar would be appear in the middle of the screen.
app.layout = dcc.Loading(
    id = 'loading_page_content',
    children = [
        html.Div(
            [   # Variable Settings to share over pages
                dcc.Store(id='input-dict', data={}),
                navbar,
                dash.page_container
            ]
        )
    ],
    color = 'primary',
    fullscreen = True
)



# Run the app
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)        # For debugging
    app.run(debug=True, host="0.0.0.0",port="3030")