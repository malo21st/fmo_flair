from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
import pandas as pd
# from jupyter_dash import JupyterDash
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

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.YETI], # SUPERHERO BOOTSTRAP
                )

# View
title = html.Div([
    dbc.Row([
        dbc.Col(
            dbc.Alert(html.H3("構造式　一覧"), 
                      color="primary", style={'text-align':'center'}
                      )
        ),
    ], justify="center")
])

select_btn = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Button("1", color="primary", id={'type':'my-button', 'index':1}, outline=True, className="btn btn-lg btn-block"),
            dcc.Input(id={'type':'my-txt', 'index':1}, style={"width": '100%'}),
        ], width=1),
        dbc.Col([
            dbc.Button("2", color="primary", id={'type':'my-button', 'index':2}, outline=True, className="btn btn-lg btn-block"),
            dcc.Input(id={'type':'my-txt', 'index':2}, style={"width": '100%'}),
        ], width=1),
        dbc.Col([
            dbc.Button("3", color="primary", id={'type':'my-button', 'index':3}, outline=True, className="btn btn-lg btn-block"),
            dcc.Input(id={'type':'my-txt', 'index':3}, style={"width": '100%'}),
        ], width=1),
        dbc.Col([
            dbc.Button("4", color="primary", id={'type':'my-button', 'index':4}, outline=True, className="btn btn-lg btn-block"),
            dcc.Input(id={'type':'my-txt', 'index':4}, style={"width": '100%'}),
        ], width=1),
        dbc.Col([
            dbc.Button("5", color="primary", id={'type':'my-button', 'index':5}, outline=True, className="btn btn-lg btn-block"),
            dcc.Input(id={'type':'my-txt', 'index':5}, style={"width": '100%'}),
        ], width=1),
        dbc.Col([
            dbc.Button("6", color="primary", id={'type':'my-button', 'index':6}, outline=True, className="btn btn-lg btn-block"),
            dcc.Input(id={'type':'my-txt', 'index':6}, style={"width": '100%'}),
        ], width=1),
        dbc.Col([
            dbc.Button("7", color="primary", id={'type':'my-button', 'index':7}, outline=True, className="btn btn-lg btn-block"),
            dcc.Input(id={'type':'my-txt', 'index':7}, style={"width": '100%'}),
        ], width=1),
        dbc.Col([
            dbc.Button("8", color="primary", id={'type':'my-button', 'index':8}, outline=True, className="btn btn-lg btn-block"),
            dcc.Input(id={'type':'my-txt', 'index':8}, style={"width": '100%'}),
        ], width=1),
        dbc.Col([
            dbc.Button("9", color="primary", id={'type':'my-button', 'index':9}, outline=True, className="btn btn-lg btn-block"),
            dcc.Input(id={'type':'my-txt', 'index':9}, style={"width": '100%'}),
        ], width=1),
        dbc.Col([
            dbc.Button("10", color="primary", id={'type':'my-button', 'index':10}, outline=True, className="btn btn-lg btn-block"),
            dcc.Input(id={'type':'my-txt', 'index':10}, style={"width": '100%'}),
        ], width=1),
        dbc.Col([
            dbc.Button("11", color="primary", id={'type':'my-button', 'index':11}, outline=True, className="btn btn-lg btn-block"),
            dcc.Input(id={'type':'my-txt', 'index':11}, style={"width": '100%'}),
        ], width=1),
        dbc.Col([
            dbc.Button("12", color="primary", id={'type':'my-button', 'index':12}, outline=True, className="btn btn-lg btn-block"), 
            dcc.Input(id={'type':'my-txt', 'index':12}, style={"width": '100%'}),
        ], width=1),
    ], no_gutters=True) #, justify="bitween")
], style={'margin':'0px 0px 10px 0px'}) # マージン：上　右　下　左

cards = dbc.Row(id="cards", justify="start",)

app.layout = dbc.Container([
                title, 
                select_btn,
                html.Div([
                    html.P(id='class_num'),
                    html.P(id='monitor'),
                # ], style = {'display': 'block'}), 
                ], style = {'display': 'none'}), 
                html.Hr(),
                cards,
                # 変数・コントロール・デバッグ用
                dcc.Store(data=dict(), id='data'),
], fluid=False) #, style={'margin':'20px 0px 0px 0px'}) # マージン：上　右　下　左


def mol_img_from_ID(num, width, height, dic_data):
    try:
        mol = Chem.MolFromSmiles(dic_data['PUBCHEM_OPENEYE_CAN_SMILES'][str(num)])
        img = Draw.MolToImage(mol, size=(width,height))
    except:
        img = "Error"
    return img

def card(num, cls, dic_data):
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
    ]
    return card_content

# Controller
@app.callback(
    Output('data', 'data'), 
    Output('cards', 'children'),
    # Output('monitor', 'children'),
    [Input('class_num', 'children')],
    [State('data', 'data'),
     State({'type': 'check', 'index': ALL}, 'value'),]
)
# Model
def store_data(class_num, dic_data, check):
    N = 23
    if dic_data == {}:
        df = PandasTools.LoadSDF('PubChem_1k.sdf')
        df = df[['ID','PUBCHEM_OPENEYE_CAN_SMILES']]
        df = df.head(N)
        df['GROUP']=3
        # df['GROUP']=[random.randint(1,12) for i in range(N)]
        jsn = df.to_json(date_format='iso')
        dic_data = ast.literal_eval(jsn)
    else:
        for flag in check:
            if len(flag):
                dic_data['GROUP'][str(flag[0])] = int(class_num) 
    # 構造式一覧
    cards = [dbc.Col([dbc.Card(card(num, dic_data['GROUP'][str(num)], dic_data), color="light", inverse=False), html.Hr()], lg=2, md=3, sm=4, xs=4) for num in dic_data['ID'].keys()]

    return dic_data, cards #, str(check)

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
        clicked_id_text = ctx.triggered[0]['prop_id'].split('.')[0]  # '{"index":x,"type":"button"}'
        clicked_id_dic = ast.literal_eval(clicked_id_text)  # evalは恐ろしいのでastを使おう
        clicked_index = clicked_id_dic['index'] # xを取り出す

    return clicked_index

app.run_server(host='0.0.0.0', port=8001, debug=True)