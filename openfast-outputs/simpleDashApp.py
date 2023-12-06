from dash import Dash, Input, Output, callback, dcc, html
import copy
import plotly.express as px
import pandas as pd

df = pd.read_csv('of-output/NREL5MW_OC3_spar_0.out', skiprows=[0,1,2,3,4,5,7], delim_whitespace=True)


app = Dash(__name__)

app.layout = html.Div(
    [
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


@callback(
    Output("line", "figure"),
    Input("signalx", "value"),
    Input("signaly", "value"),
)
def line_chart(signalx,signaly):

    fig = px.line(
        x=df[signalx],
        y=df[signaly],
        template="simple_white",
        labels={"x": signalx, "y": signaly},
    )

    return fig


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",port="3030")
 
