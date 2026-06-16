import pandas as pd
from sklearn.preprocessing import RobustScaler
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# ----------------------------------------
# Global Preprocessing
# ----------------------------------------
df = pd.read_csv('creditcard.csv')

scalar = RobustScaler()
df['Scaled_Time'] = scalar.fit_transform(df['Time'].values.reshape(-1,1))
df['Scaled_Amount'] = scalar.fit_transform(df['Amount'].values.reshape(-1,1))
df.drop(['Time','Amount'], axis=1, inplace=True)

df_cn = df[df['Class'] == 0]
df_fraud = df[df['Class'] == 1]

total_txn = len(df)
fraud_txn = df_fraud.shape[0]
fraud_pct = 100 * fraud_txn / total_txn

features = [col for col in df.columns if col != 'Class']

# ----------------------------------------
# App Initialization
# ----------------------------------------
app = Dash(__name__)
server = app.server
app.title = "Credit Card Fraud Detection Dashboard"

# ----------------------------------------
# App Layout
# ----------------------------------------
app.layout = html.Div([
    html.H1("💳 Credit Card Fraud Detection Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.H3("Total Transactions"),
            html.H2(f"{total_txn:,}")
        ], className="card"),

        html.Div([
            html.H3("Fraudulent Transactions"),
            html.H2(f"{fraud_txn:,}")
        ], className="card"),

        html.Div([
            html.H3("Fraud Percentage"),
            html.H2(f"{fraud_pct:.3f}%")
        ], className="card"),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),

    html.Hr(),

    html.Div([
        html.Div([
            html.Label("Select X-axis Feature:"),
            dcc.Dropdown(features, value='V10', id='x_feature', clearable=False)
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Select Y-axis Feature:"),
            dcc.Dropdown(features, value='V14', id='y_feature', clearable=False)
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ], style={'margin': '10px'}),

    html.Hr(),

    html.Div([
        html.H3("Scatter Plot (Feature Relationships)"),
        dcc.Graph(id='scatter_plot')
    ]),

    html.Div([
        html.H3("Histogram (Feature Distribution)"),
        html.Label("Select Feature for Histogram:"),
        # Fixed default value from 'Amount' to 'Scaled_Amount'
        dcc.Dropdown(features, value='Scaled_Amount', id='hist_feature', clearable=False),
        dcc.Graph(id='hist_plot')
    ]),

    html.Div([
        html.H3("Box Plot (Feature vs Class)"),
        html.Label("Select Feature for Box Plot:"),
        dcc.Dropdown(features, value='V12', id='box_feature', clearable=False),
        dcc.Graph(id='box_plot')
    ]),

    html.Div([
        html.H3("Violin Plot (Feature vs Class)"),
        html.Label("Select Feature for Violin Plot:"),
        dcc.Dropdown(features, value='V14', id='violin_feature', clearable=False),
        dcc.Graph(id='violin_plot')
    ])
])

# ----------------------------------------
# Callbacks
# ----------------------------------------
@app.callback(
    Output('scatter_plot', 'figure'),
    [Input('x_feature', 'value'),
     Input('y_feature', 'value')]
)
def update_scatter(x_feature, y_feature):
    fig = px.scatter(
        df, x=x_feature, y=y_feature, color='Class',
        color_discrete_map={0: 'blue', 1: 'red'},
        opacity=0.6,
        title=f'{x_feature} vs {y_feature}',
        labels={'Class': 'Transaction Class'}
    )
    fig.update_layout(template='plotly_dark')
    return fig

@app.callback(
    Output('hist_plot', 'figure'),
    [Input('hist_feature', 'value')]
)
def update_histogram(feature):
    # Fixed df_merge to df
    fig = px.histogram(
        df, x=feature, color='Class',
        barmode='overlay', opacity=0.6,
        color_discrete_map={0: 'blue', 1: 'red'},
        title=f'Distribution of {feature}'
    )
    fig.update_layout(template='plotly_white')
    return fig

@app.callback(
    Output('box_plot', 'figure'),
    [Input('box_feature', 'value')]
)
def update_boxplot(feature):
    fig = px.box(
        df, x='Class', y=feature, color='Class',
        color_discrete_map={0: 'blue', 1: 'red'},
        title=f'Box Plot of {feature} by Class'
    )
    fig.update_layout(template='plotly_white')
    return fig

@app.callback(
    Output('violin_plot', 'figure'),
    [Input('violin_feature', 'value')]
)
def update_violin(feature):
    fig = px.violin(
        df, x='Class', y=feature, color='Class',
        box=True, points='all',
        color_discrete_map={0: 'blue', 1: 'red'},
        title=f'Violin Plot of {feature} by Class'
    )
    fig.update_layout(template='plotly_white')
    return fig

# ----------------------------------------
# Execution Block
# ----------------------------------------
if __name__ == '__main__':
    # Cleaned up to a single execution call for a .py script
    app.run(debug=False, port=8050)