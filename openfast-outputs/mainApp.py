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
        dbc.DropdownMenu(
            [dbc.DropdownMenuItem('Blade', href='/wisdem_blade'), dbc.DropdownMenuItem('Cost', href='/wisdem_cost')],
            label="WISDEM",
            nav=True
        ),
        # dbc.NavItem(dbc.NavLink("WISDEM", href='/wisdem')),
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

# might have to move this to viz utils
def checkPort(port, host="0.0.0.0"):
    import socket
    # check port availability and then close the socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = False
    try:
        sock.bind((host, port))
        result = True
    except:
        result = False

    sock.close()
    return result


# Run the app
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='WEIS Visualization App')
    parser.add_argument('--port', type=int, default=8050, help='Port number to run the app')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host IP to run the app')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')

    args = parser.parse_args()

    # test the port availability, flask calls the main function twice in debug mode
    if not checkPort(args.port, args.host) and not args.debug:
        print(f"Port {args.port} is already in use. Please change the port number and try again.")
        print(f"To change the port number, pass the port number with the '--port' flag. eg: python mainApp.py --port {args.port+1}")
        print("Exiting the app.")
        exit()

    logging.basicConfig(level=logging.DEBUG)        # For debugging
    app.run(debug=args.debug, host=args.host, port=args.port)

