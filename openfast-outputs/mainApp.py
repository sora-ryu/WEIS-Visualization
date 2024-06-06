'''Main Page where we get the input file'''

# Import Packages
import dash
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
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
        dbc.NavItem(dbc.NavLink("Optimization", href='/optimize')),
        dbc.DropdownMenu(
            [dbc.DropdownMenuItem('Blade', href='/wisdem_blade'), dbc.DropdownMenuItem('Cost', href='/wisdem_cost'), dbc.DropdownMenuItem('General', href='/wisdem_general')],
            label="WISDEM",
            nav=True
        )
    ],
    brand = APP_TITLE,
    color = "darkblue",
    dark = True,
    className = "menu-bar"
)

# Wrap app with loading component
# Whenever it needs some time for loading data, small progress bar would be appear in the middle of the screen.
file_indices = ['file1', 'file2']       # Need to define as the way defined in .yaml file

app.layout = dcc.Loading(
    id = 'loading_page_content',
    children = [
        html.Div(
            [   # Variable Settings to share over pages
                dcc.Store(id='input-dict', data={}),
                # OpenFAST related Data fetched from input-dict
                dcc.Store(id='var-openfast', data={}),
                dcc.Store(id='var-openfast-graph', data={}),
                # Dataframe to share over functions - openfast .out file
                 html.Div(
                    [dcc.Store(id=f'df-{idx}', data={}) for idx in file_indices]      # dcc.Store(id='df-file1', data={}),          # {file1, df1}
                ),
                # Optimization related Data fetched from input-dict
                dcc.Store(id='var-opt', data={}),
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

