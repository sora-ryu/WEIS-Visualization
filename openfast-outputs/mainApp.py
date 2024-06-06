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



# Run the app
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)        # For debugging
    app.run(debug=True, host="0.0.0.0",port="3030")