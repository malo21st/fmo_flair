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
            dbc.Alert(html.H3("手動振分けデモ"), 
                      color="primary", style={'text-align':'center'}
                      )
        ),
    ], justify="center")
])

data_and_card = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Button(
                "Load Data", color="primary", id={'type':'my-button', 'index':0}, outline=True, # id={'type':'my-button', 'index':0}
                className="btn btn-lg btn-block"
            ),
        ], width=4),
        dbc.Col([
            dcc.Loading(id="id_loading",
                children=[
                    dbc.Card([
                        dbc.CardHeader(id='mol-num'),
                        dbc.CardBody(
                            html.Img(id='mol-img')
                        ),
                    ]),
                ],
            ),
        ], width=4),
    ], justify="start")
], style={'margin':'20px 0px 20px 0px'}) # マージン：上　右　下　左

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

latest = html.Div([
    dbc.Row([
        dbc.Col(html.Img(id={'type':'class', 'index':1}), width=1),
        dbc.Col(html.Img(id={'type':'class', 'index':2}), width=1),
        dbc.Col(html.Img(id={'type':'class', 'index':3}), width=1),
        dbc.Col(html.Img(id={'type':'class', 'index':4}), width=1),
        dbc.Col(html.Img(id={'type':'class', 'index':5}), width=1),
        dbc.Col(html.Img(id={'type':'class', 'index':6}), width=1),
        dbc.Col(html.Img(id={'type':'class', 'index':7}), width=1),
        dbc.Col(html.Img(id={'type':'class', 'index':8}), width=1),
        dbc.Col(html.Img(id={'type':'class', 'index':9}), width=1),
        dbc.Col(html.Img(id={'type':'class', 'index':10}), width=1),
        dbc.Col(html.Img(id={'type':'class', 'index':11}), width=1),
        dbc.Col(html.Img(id={'type':'class', 'index':12}), width=1),
    ], no_gutters=True)
], style={'margin':'0px 0px 0px 0px'}) # マージン：上　右　下　左

more_btn = html.Div([
    dbc.Row([
        dbc.Col(dbc.Button(0, color="primary", id={'type':'more', 'index':1}, disabled=True, size='sm'), width=1),
        dbc.Col(dbc.Button(0, color="primary", id={'type':'more', 'index':2}, disabled=True, size='sm'), width=1),
        dbc.Col(dbc.Button(0, color="primary", id={'type':'more', 'index':3}, disabled=True, size='sm'), width=1),
        dbc.Col(dbc.Button(0, color="primary", id={'type':'more', 'index':4}, disabled=True, size='sm'), width=1),
        dbc.Col(dbc.Button(0, color="primary", id={'type':'more', 'index':5}, disabled=True, size='sm'), width=1),
        dbc.Col(dbc.Button(0, color="primary", id={'type':'more', 'index':6}, disabled=True, size='sm'), width=1),
        dbc.Col(dbc.Button(0, color="primary", id={'type':'more', 'index':7}, disabled=True, size='sm'), width=1),
        dbc.Col(dbc.Button(0, color="primary", id={'type':'more', 'index':8}, disabled=True, size='sm'), width=1),
        dbc.Col(dbc.Button(0, color="primary", id={'type':'more', 'index':9}, disabled=True, size='sm'), width=1),
        dbc.Col(dbc.Button(0, color="primary", id={'type':'more', 'index':10}, disabled=True, size='sm'), width=1),
        dbc.Col(dbc.Button(0, color="primary", id={'type':'more', 'index':11}, disabled=True, size='sm'), width=1),
        dbc.Col(dbc.Button(0, color="primary", id={'type':'more', 'index':12}, disabled=True, size='sm'), width=1),
    ], no_gutters=True, justify="center")
], style={'margin':'10px 0px 0px 0px'}) # マージン：上　右　下　左

app.layout = dbc.Container([
                            title, 
                            data_and_card,
                            select_btn,
                            latest,
                            more_btn,
                            # 変数・コントロール・デバッグ用
                            dcc.Store(id='data'),
                            dcc.Store(id='index'),
                            dcc.Store(id={'type':'classify', 'index':1}, data=[]),
                            dcc.Store(id={'type':'classify', 'index':2}, data=[]),
                            dcc.Store(id={'type':'classify', 'index':3}, data=[]),
                            dcc.Store(id={'type':'classify', 'index':4}, data=[]),
                            dcc.Store(id={'type':'classify', 'index':5}, data=[]),
                            dcc.Store(id={'type':'classify', 'index':6}, data=[]),
                            dcc.Store(id={'type':'classify', 'index':7}, data=[]),
                            dcc.Store(id={'type':'classify', 'index':8}, data=[]),
                            dcc.Store(id={'type':'classify', 'index':9}, data=[]),
                            dcc.Store(id={'type':'classify', 'index':10}, data=[]),
                            dcc.Store(id={'type':'classify', 'index':11}, data=[]),
                            dcc.Store(id={'type':'classify', 'index':12}, data=[]),
                            html.Div([
                                html.Label(id={'type':'debug', 'index':1}),
                                html.Label(id={'type':'debug', 'index':2}),
                                html.Label(id={'type':'debug', 'index':3}),
                                html.Label(id={'type':'debug', 'index':4}),
                                html.Label(id={'type':'debug', 'index':5}),
                                html.Label(id={'type':'debug', 'index':6}),
                                html.Label(id={'type':'debug', 'index':7}),
                                html.Label(id={'type':'debug', 'index':8}),
                                html.Label(id={'type':'debug', 'index':9}),
                                html.Label(id={'type':'debug', 'index':10}),
                                html.Label(id={'type':'debug', 'index':11}),
                                html.Label(id={'type':'debug', 'index':12}),
                                html.Label(id='monitor'),
                # ], style = {'display': 'block'}), 
                ], style = {'display': 'none'}), 
], fluid=False)


# # Controller
# @app.callback(
#     Output({'type': 'debug', 'index': MATCH}, 'children'), 
#     [Input({'type': 'more', 'index': MATCH}, 'n_clicks')],
#     State({'type': 'classify', 'index': MATCH}, 'data'),
#     prevent_initial_call=True,
# )
# # Model
# def class_list(n_clicks, lst_class):
#     ctx = dash.callback_context
#     if not ctx.triggered or ctx.triggered[0]['value'] is None:
#         raise PreventUpdate
#     else:
#         clicked_id_text = ctx.triggered[0]['prop_id'].split('.')[0]  # '{"index":x,"type":"button"}'
#         clicked_id_dic = ast.literal_eval(clicked_id_text)  # evalは恐ろしいのでastを使おう
#         clicked_index = clicked_id_dic['index'] # xを取り出す

#         return f'　{clicked_index} : {str(lst_class)},'

def mol_img_from_ID(num, width, height, dic_data):
    try:
        mol = Chem.MolFromSmiles(dic_data['PUBCHEM_OPENEYE_CAN_SMILES'][str(num)])
        img = Draw.MolToImage(mol, size=(width,height))
    except:
        img = "Error"
    return img

# Controller
@app.callback(
    Output({'type': 'class', 'index': MATCH}, "src"), 
    Output({'type': 'classify', 'index': MATCH}, "data"),
    Output({'type': 'more', 'index': MATCH}, "children"),
    Output({'type': 'more', 'index': MATCH}, "disabled"),
    [Input({'type': 'my-button', 'index': MATCH}, 'n_clicks')],
    [State("data", "data"), State("index", "data"), 
     State({'type': 'classify', 'index': MATCH}, "data")],
    prevent_initial_call=True,
)
# Model
def classification(n_clicks, dic_data, data_idx, lst_class):
    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['value'] is None:
        raise PreventUpdate
    else:
        clicked_id_text = ctx.triggered[0]['prop_id'].split('.')[0]  # '{"index":x,"type":"button"}'
        clicked_id_dic = ast.literal_eval(clicked_id_text)  # evalは恐ろしいのでastを使おう
        clicked_index = clicked_id_dic['index'] # xを取り出す

        if clicked_index == 0:
            lst_class = []
            img = None
        else:
            lst_class.append(data_idx)
            img = mol_img_from_ID(data_idx, 90, 60, dic_data)
            len_lst = len(lst_class)
            if len_lst:
                disabled = False
            else:
                disabled = True
        return img, lst_class, len_lst, disabled

# Controller
@app.callback(
    Output("data", "data"),
    Output("index", "data"),
    Output("mol-num", "children"), 
    Output("mol-img", "src"),
    # Output("monitor", "children"), 
    [Input({'type': 'my-button', 'index': ALL}, 'n_clicks')],
    [State("data", "data"), State("index", "data")],
    prevent_initial_call=True,
)
# Model
def click_button(n_clicks, dic_data, index):
    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['value'] is None:
        return 'No clicks yet'
    else:
        clicked_id_text = ctx.triggered[0]['prop_id'].split('.')[0]  # '{"index":x,"type":"button"}'
        clicked_id_dic = ast.literal_eval(clicked_id_text)  # evalは恐ろしいのでastを使おう
        clicked_index = clicked_id_dic['index'] # xを取り出す

        if clicked_index == 0:
            df = PandasTools.LoadSDF('PubChem_1k.sdf')
            df = df[['ID','PUBCHEM_OPENEYE_CAN_SMILES']]
            jsn = df.to_json(date_format='iso')
            dic_data = ast.literal_eval(jsn)
            index = 0
        else:
            index += 1

        len_data = len(dic_data['ID'])
        number = index + 1
        img = mol_img_from_ID(index, 300, 200, dic_data)

        return dic_data, index, f'{number:>8} / {len_data:,}', img #, clicked_index

app.run_server(host='0.0.0.0', port=8001, debug=True)