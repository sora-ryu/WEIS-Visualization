# Import Packages
from dash import Dash, Input, Output, callback, dcc, html, dash_table
import copy
import plotly.express as px
import pandas as pd


# Incorporate Data
df = pd.read_csv('of-output/NREL5MW_OC3_spar_0.out', skiprows=[0,1,2,3,4,5,7], delim_whitespace=True)

# Initialize the app - Internally starts the Flask Server
app = Dash(__name__)

# App Layout
app.layout = html.Div(
    [
        html.Div(children='NREL5MW_OC3_spar_0 data'),
        dash_table.DataTable(data=df.to_dict('records'), page_size=10),

        html.H5("Signal-y"),
        dcc.Dropdown(
            id="signaly", options=sorted(df.keys())
        ),
        html.H5("Signal-x"),
        dcc.Dropdown(
            id="signalx", options=sorted(df.keys())
        ),
        html.Br(),
        dcc.Graph(id="line"),
    ]
)

# Add controls to build the interaction
@callback(
    Output(component_id="line", component_property="figure"),
    Input(component_id="signalx", component_property="value"),
    Input(component_id="signaly", component_property="value"),
)

def line_chart(signalx,signaly):

    fig = px.line(
        x=df[signalx],
        y=df[signaly],
        template="simple_white",
        labels={"x": signalx, "y": signaly},
    )

    return fig


# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",port="3030")
 
