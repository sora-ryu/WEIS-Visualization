'''This is the page for visualizing table and plots'''

# Import Packages
from dash import Dash, Input, Output, callback, dcc, html, dash_table
import copy
import plotly.express as px
import plotly.graph_objects as go
import plotly.tools as tls
from plotly.subplots import make_subplots
import dash_mantine_components as dmc
import pandas as pd


# Incorporate Data
data_path = 'of-output/'
data_file = 'NREL5MW_OC3_spar_0.out'
df = pd.read_csv(data_path + data_file, skiprows=[0,1,2,3,4,5,7], delim_whitespace=True)

# Initialize the app - Internally starts the Flask Server
# Incorporate a Dash Mantine theme
external_stylesheets = [dmc.theme.DEFAULT_COLORS]
app = Dash(__name__, external_stylesheets = external_stylesheets)

# App Layout
app.layout = html.Div(
    [
        html.Div(data_file.split('.')[-2]),
        dash_table.DataTable(data=df.to_dict('records'), page_size=10),

        html.H5("Signal-y"),
        dcc.Dropdown(
            id="signaly", options=sorted(df.keys()), multi=True
        ),
        html.H5("Signal-x"),
        dcc.Dropdown(
            id="signalx", options=sorted(df.keys())
        ),
        html.Br(),
        html.H5("Plot Options"),
        dcc.RadioItems(
            options=['single plot', 'multiple plot'], value='single plot', inline=True, id='plotOption'
        ),
        html.Br(),
        dcc.Graph(id="line"),
    ]
)

# Add function to get the data from POST request
def get_file_path(request):
    if request.method == 'POST':
        post_data = json.loads(request.body.decode("utf-8"))
        value = post_data.get('data')
        print(value)




# Add controls to build the interaction
@callback(
    Output(component_id="line", component_property="figure"),
    Input(component_id="signalx", component_property="value"),
    [Input(component_id="signaly", component_property="value")],
    Input(component_id="plotOption", component_property="value")
)
def draw_graphs(signalx, signaly, plotOption):

    # fig = tls.make_subplots(rows=1, cols=1, shared_xaxes=True, verical_spacing=0.009, horizontal_spacing=0.009)
    # fig = go.Figure()

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

    
    return fig


# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",port="3030")
 