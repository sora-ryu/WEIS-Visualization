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
import dash_bootstrap_components as dbc
import pandas as pd
import logging


# Initialize the app - Internally starts the Flask Server
# Incorporate a Dash Mantine theme
# external_stylesheets = [dmc.theme.DEFAULT_COLORS]
external_stylesheets = [dbc.themes.BOOTSTRAP]
APP_TITLE = "WEIS Visualization APP"
app = Dash(__name__, external_stylesheets = external_stylesheets, suppress_callback_exceptions=True, title=APP_TITLE, use_pages=True)

# Build Simple Navigation Bar
navbar = dbc.NavbarSimple(
    children = [
        dbc.NavItem(dbc.NavLink("Home", href='/')),
        dbc.NavItem(dbc.NavLink("OpenFAST", href='/open_fast')),
        dbc.NavItem(dbc.NavLink("Optimize", href='/optimize')),
        dbc.NavItem(dbc.NavLink("3D Visualization", href='/3d_vis'))
    ],
    brand = APP_TITLE,
    color = "dark",
    dark = True,
    className = "menu-bar"
)

# Wrap app with loading component
app.layout = dcc.Loading(
    id = 'loading_page_content',
    children = [
        html.Div(
            [
                dcc.Store(id='store', data={}),
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