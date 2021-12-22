from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL, MATCH, ALLSMALLER
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import base64
import datetime
import io
import ast
import random

# df = pd.read_csv("Kinase_p38.csv", header=0)
# df = PandasTools.LoadSDF("PubChem_1k.sdf")

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.YETI], # SUPERHERO BOOTSTRAP
                )

# View
# the style arguments for the sidebar. We use position:fixed and a fixed width
HEADDER_STYLE = {
    "zIndex": 30,
    "position": "fixed",
    # "top": 0,
    "left": "3rem",
    "right":"4rem",
    # "bottom": 0,
    # "width": "10rem",
    # "margin-right": "2rem",
    # "padding": "0rem 0rem",
    "background-color": "#f8f9fa",
}

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "zIndex": 20,
    "position": "fixed",
    "top": 190,
    "left": 0,
    "bottom": 0,
    "width": "10rem",
    "margin-left": "1rem",
    "padding": "0rem 0rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "zIndex": 10,
    "position": "absolute",
    "top": 190,
    "left": 180,
    "margin-left": "0rem",
    "margin-right": "5rem",
    "padding": "0rem 0rem",
}

title = html.Div([
    dbc.Row([
        dbc.Col(
            dbc.Alert(html.H3("構造式　一覧"), 
                      color="primary", style={'text-align':'center'}
                      )
        ),
    ], justify="center")
])

select_row = html.Div([
    dbc.Row(id='select-row', no_gutters=True) #, justify="bitween")
], style={'margin':'0px 0px 10px 0px'}) # マージン：上　右　下　左

cards = html.Div(dbc.Row(id="cards", justify="start",), style=CONTENT_STYLE)

sidebar = html.Div(
    [
        html.P("表示項目", className="lead"),
        # html.P("(max:5)", className="lead"),
        dcc.Checklist(
            id='check-item',
            value=[],
            labelStyle = {'display':'block'}
        ),
    ],style=SIDEBAR_STYLE,
)

# Controller
@app.callback(
    Output('check-item', 'options'), 
    [Input('check-item', 'options')],
)
# Model
def check_item(op):
    df = pd.read_csv("Kinase_p38.csv", header=0)
    df = df.dropna(how='all').dropna(how='all', axis=1)
    options = [{'label': f"{item}", 'value': item} for item in df.columns]
    return options


app.layout = dbc.Container([
                html.Div([
                    title, 
                    select_row,
                    html.Hr(),
                ], style = HEADDER_STYLE),
                sidebar,
                cards,
                # 変数・コントロール・デバッグ用
                html.Div([
                    html.P(id='class_num'),
                    html.P(id='monitor'),
                # ], style = {'display': 'block'}), # 表示
                ], style = {'display': 'none'}), # 非表示
                dcc.Store(data=dict(), id='data'),
], fluid=False) #, style={'margin':'20px 0px 0px 0px'}) # マージン：上　右　下　左

def make_btn(num):
    col = dbc.Col([
            dbc.Button(f"{num}", color="primary", id={'type':'my-button', 'index':num}, outline=True, className="btn btn-lg btn-block"),
            dcc.Input(id={'type':'my-txt', 'index':num}, style={"width": '100%'}),
        ], width=1)
    return col

# Controller
@app.callback(
    Output('select-row', 'children'), 
    [Input('select-row', 'children')],
)
# Model
def select_row(col):
    num = 12
    return [make_btn(i) for i in range(1, num+1)]

def mol_img_from_ID(num, width, height, dic_data):
    try:
        mol = Chem.MolFromSmiles(dic_data['CANONICAL_SMILES'][str(num)])
        img = Draw.MolToImage(mol, size=(width,height))
    except:
        img = "Error"
    return img

def card(num, cls, dic_data, check_item):
    card_content = [
            dbc.CardHeader([
                dcc.Checklist(
                    options=[
                        {'label': f'  [ {cls} ]　{num}', 'value': num},
                    ],
                    value=[],
                    id={'type':'check', 'index':num},
                ),
            ]),
            dbc.CardImg(src = mol_img_from_ID(num, 150, 150, dic_data), top=True, bottom=False),
            dbc.CardBody([html.Div([html.B(html.Label(f"{item}：")), html.Label(f"{dic_data[item][str(num)]}")]) for item in check_item]),
    ]
    return card_content

# Controller
@app.callback(
    Output('data', 'data'), 
    Output('cards', 'children'),
    Output('monitor', 'children'),
    [Input('class_num', 'children'),
     Input('check-item', 'value')],
    [State('data', 'data'),
     State({'type': 'check', 'index': ALL}, 'value'),]
)
# Model
def store_data(class_num, check_item, dic_data, check):
    N = 30
    if dic_data == {}:
        # df = PandasTools.LoadSDF('PubChem_1k.sdf')
        df = pd.read_csv("Kinase_p38.csv", header=0)
        df = df.head(N)
        df = df.dropna(how='all').dropna(how='all', axis=1)
        # df = df[['ID','PUBCHEM_OPENEYE_CAN_SMILES']]
        df['GROUP']=3
        # df['GROUP']=[random.randint(1,12) for i in range(N)]
        jsn = df.to_json(date_format='iso')
        dic_data = eval(jsn)
    else:
        for flag in check:
            if len(flag):
                dic_data['GROUP'][str(flag[0])] = int(class_num) 
    # 構造式一覧
    check_item = [item for item in dic_data.keys() if item in check_item]
    cards = [dbc.Col([
                dbc.Card(
                    card(num, dic_data['GROUP'][str(num)], dic_data, check_item), 
                    color="light", inverse=False
                    ),
                html.P(),
                ], lg=2, md=3, sm=4, xs=4
            ) for num in dic_data['Index'].keys()]

    return dic_data, cards , str(check_item)

# Controller
@app.callback(
    Output('class_num', 'children'), 
    [Input({'type': 'my-button', 'index': ALL}, 'n_clicks')],
    State('data', 'data'),
    prevent_initial_call=True,
)
# Model
def class_list(n_clicks, dic_data): #, check):
    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['value'] is None:
        raise PreventUpdate
    else:
        clicked_id_text = ctx.triggered[0]['prop_id'].split('.')[0]
        clicked_id_dic = eval(clicked_id_text)
        clicked_index = clicked_id_dic['index']

    return clicked_index

app.run_server(host='0.0.0.0', port=8001, debug=True)