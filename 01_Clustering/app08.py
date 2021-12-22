from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools, rdMolInterchange, rdCoordGen
from rdkit.Chem.Draw import rdMolDraw2D, IPythonConsole
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, ALL, MATCH, ALLSMALLER
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import base64
import datetime
import io
import time

from flask_caching import Cache
import uuid

SLEEP_TIME = 1 # time.sleep(SLEEP_TIME)
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.YETI], # SUPERHERO BOOTSTRAP
                suppress_callback_exceptions=True,
                )

# View

class_dic ={'ward':'ウォード法', 'average':'平均連結法', 'complete':'完全連結法'}

def table_df(file_name, records, columns):
    df = pd.DataFrame(
        {
            "項　目": ["ファイル名", "レコード数", "項　目　数"],
            "内　　　容": [f"{file_name}", f"{records}", f"{columns}"],
        }
    )
    return df

title = html.Div([
    dbc.Row([
        dbc.Col(
            dbc.Alert(html.H3("fmo-flair プロトタイプ"), 
                      color="primary", style={'text-align':'center'}
                      )
        ),
    ], justify="center")
])

class_row = html.Div([
    dbc.Row([
        dbc.Col(
            dcc.Upload(
                id='upload-data',
                children=html.Div(
                    ['Drag and Drop SDF',
                    'or',
                    html.A('Select Files')
                    ]),
                style={
                    'width': '80%',
                    'height': '100%',
                    'lineHeight': '80px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                    },
                multiple=False,
            ), width=3
        ),
        dbc.Col(
            dcc.Loading(
                id='loading',
                type='default',
                children=[
                    dbc.Table(
                        # dbc.Table.from_dataframe(table_df("", "", ""), striped=True, bordered=True, hover=True), 
                        id="data-table",
                    )
                ]
            ), width=6
        ),
        dbc.Col([
            html.Div([
                html.H6('方　法：', style={'display': 'inline-block', 'width':'25%', 'vertical-align':'middle'}),
                dcc.Dropdown(
                    id='cluster-arg',
                    options=[{'label': item, 'value': key} for key, item in class_dic.items()],
                    clearable=False, 
                    persistence=False, 
                    persistence_type='local',
                    style={'display': 'inline-block', 'width':'75%', 'vertical-align':'middle'},
                ),
            ], style={'margin':'0px 0px 10px 0px'}),
            html.Div([
                html.H6('分割数：', style={'display': 'inline-block', 'width':'25%', 'vertical-align':'middle'}),
                dcc.Dropdown(
                    id='cluster-num',
                    options=[{'label': item, 'value': item} for item in range(2,11)],
                    clearable=False, 
                    persistence=False, 
                    persistence_type='local',
                    style={'display': 'inline-block', 'width':'75%', 'vertical-align':'middle'},
                ),
            ], style={'margin':'0px 0px 10px 0px'}),
            html.Div([
                dbc.Button('　実　行　', id="exe-btn", color="primary", disabled=True, className="btn btn-md"),
            ], style={'margin':'0px 50px 10px 0px', 'text-align':'right'})
        ], width=3) #, style={'text-align':'left'}, align='start'),
    ], no_gutters=False, justify="start")
], style={'margin':'10px 10px 0px 10px'}) # マージン：上　右　下　左


def mol_img(num, width, height, mols_data):
    try:
        mols = [rdMolInterchange.JSONToMols(jsn) for jsn in mols_data]
        mol = mols[num][0]
        mol.RemoveAllConformers()
        rdCoordGen.AddCoords(mol)
        img = Draw.MolToImage(mol, size=(width,height))
    except:
        img = "Error"
    return img

def card(num, mols_data, cls, item_data, check_item):
    card_content = [
            dbc.CardHeader([
                dcc.Checklist(
                    options=[
                        {'label': f'  [ {cls} ]　{num}', 'value': num},
                    ],
                    value=[],
                    id={'type':'check-card', 'index':num},
                ),
            ]),
            dbc.CardImg(src = mol_img(num, 150, 150, mols_data), top=True, bottom=False),
            dbc.CardBody([html.Div([html.B(html.P(f"{item}：", className="card-text")), 
                html.P(f"{item_data[item][str(num)]}", className="card-text")]) for item in check_item],
            ),
    ]
    return card_content

@app.callback(
    Output("show-mols", 'children'),
    Output({'type': "result-btn", 'index': ALL}, 'active'),
    Output('select-cls', 'data'),
    Output('debug', 'children'),
    Input({'type': "result-btn", 'index': ALL}, 'n_clicks'),
    Input('disp-mode', 'value'), Input('select-item', 'data'),
    Input('modify-cls', 'data'),
    State('cls-data', 'data'), State('mols-data', 'data'),
    State('item-data', 'data'), State('select-cls', 'data'),
    prevent_initial_call=True,
)
def show_mols(clicks, disp_mode, check_item, modify_cls, cls_data, mols_data, item_data, selected_cls):
    print('show_mols')
    time.sleep(SLEEP_TIME)
    if modify_cls:
        # select_cls = selected_cls
        raise PreventUpdate

    btn_state = [False]*len(clicks)
    mols_cnt = list()

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        now_dic = eval(trigger_id)
        now_type = now_dic['type']
        now_index = now_dic['index']
    except:
        now_type = trigger_id
        now_index = selected_cls

    select_cls = now_index
    # if now_type in ['result-btn']:
    #     select_cls = now_index
    # # else:
    #     select_cls = selected_cls

    debug = f'{now_type} {now_index} {select_cls}'

    if not cls_data:
        return list(), btn_state, 0, debug

    if now_type in ['result-btn', 'disp-mode', 'set-item']:
        mols = [i for i in range(len(cls_data)) if cls_data[i]==select_cls]

        if disp_mode == 'CARD':
            mols_cnt = [dbc.Col([
                        dbc.Card(
                            card(num, mols_data, select_cls, item_data, check_item), 
                            color="light", inverse=False,
                            style={'max-width':'100%'}
                            ),
                        html.P(),
                        ], lg=2, md=3, sm=4, xs=4
                    ) for num in mols]
        elif disp_mode == 'TABLE':
            if select_cls:
                df = pd.DataFrame({
                        '選択':[dcc.Checklist(
                                id={'type':'check-table', 'index':num}, 
                                options=[{'label':'', 'value':num}],
                                value=[]) for num in mols],
                        'No.':mols,
                        '区分':[select_cls]*len(mols),
                        '構造式':[html.Img(src=mol_img(m, 100, 100, mols_data)) for m in mols],
                    })
                for item in check_item:
                    df[item]=[item_data[item][str(num)] for num in mols]
                mols_cnt = [dbc.Col(
                                dbc.Table.from_dataframe(
                                    df, striped=False, bordered=True, hover=True,
                                    responsive=True, className="card-text",
                                ),
                            width=12)]
            else:
                mols_cnt = list()
    
    if now_type not in ['disp-mode', 'set-item']:
        if select_cls:
            btn_state[select_cls - 1] = True
    debug = f'{now_type} {now_index} {select_cls}'

    return mols_cnt, btn_state, select_cls, debug

def result_col(i):
    return dbc.Col([
            dbc.Button([f"{i}", html.Br(),
                dbc.Badge("0", id={'type':'result-num', 'index':i}, pill=True, color="warning", className="ml-1")],
                id={'type':'result-btn', 'index':i},
                color="primary",
                outline=True,
                active=False,
                disabled=True,
                className="btn btn-lg btn-block",
                ),
        ], width=1)

result_row = html.Div([
    dbc.Row(id="result-row", no_gutters=True, justify="center"),
    dbc.Row(id="show-mols", no_gutters=False, justify="start", style={'margin':'10px 0px 0px 0px'})

], style={'margin':'10px 10px 10px 10px'}) # マージン：上　右　下　左

@app.callback(
    Output("result-row", 'children'),
    Output("trigger", 'data'),
    Input("cluster-num", 'value'),
    Input("modify-cls", 'data'),
    State("trigger", 'data'),
    State("modify-cls", 'data'),
    prevent_initial_call=True,
)
def set_result(num, modify_cls, trigger, selected_cls):
    print('set_result')
    time.sleep(SLEEP_TIME)
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        now_dic = eval(trigger_id)
        now_type = now_dic['type']
        now_index = now_dic['index']
    except:
        now_type = trigger_id
        now_index = selected_cls

    if num:
        if now_type == 'modify-cls':
            if modify_cls:
                row = [result_col(i) for i in range(1, num+2)]
            else:
                row = [result_col(i) for i in range(1, num+1)]
        elif now_type == 'cluster-num':
            row = [result_col(i) for i in range(1, num+1)]
    else:
        raise PreventUpdate
    return row , trigger+1


debug_row = html.Div([
    dbc.Row([
        dbc.Col(
            html.Label(id="debug"), width=12
        )
    ], style = {'display': 'block'}), 
    # ], style = {'display': 'none'}), 
]) 

row_dic = {1:class_row, 2:result_row}

def make_item(i):
    label_dic = {1:"クラスタリング", 2:"結　　果"}
    if i==2:
        item = [
                dcc.Dropdown(
                    id='disp-mode',
                    options=[
                        {'label':'カード', 'value':'CARD'},
                        {'label':'テーブル', 'value':'TABLE'}
                    ],
                    value='CARD',
                    clearable=False,
                    disabled=True,
                    style={'width':'5rem'},
                ),
                dbc.Button(
                    "表示設定",
                    id='set-item',
                    color="primary",
                    className="btn btn-md",
                    disabled=True,
                    style={'margin':'0px 0px 0px 20px'}, # マージン：上　右　下　左
                ),
                ]
    else:
        item = list()
    return dbc.Card(
        [
            # dbc.Button(
            dbc.NavbarSimple(
                id=f"nav-bar-{i}",
                children=item,
                brand=f"{label_dic[i]}",
                color="primary",
                dark=True,
                # n_clicks=0,
                style={'text-align':'left'},
            ),
            dbc.Collapse(
                row_dic[i],
                id=f"collapse-{i}",
                is_open=True,
            ),
        ]
    )

accordion = html.Div(
    [make_item(1), make_item(2)],
)

modal_set_item = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader("表示項目"),
                dbc.ModalBody(id='select-item-list'),
                dbc.ModalFooter(
                    dbc.Button(
                        "設　定", id="close-set-item", className="ml-auto", n_clicks=0
                    )
                ),
            ],
            id="modal-set-item",
            centered=True,
            is_open=False,
        ),
    ]
)

modal_load_data = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader("エラー"),
                dbc.ModalBody("ファイルの読込みに失敗しました。"),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="close-1", className="ml-auto", n_clicks=0
                    )
                ),
            ],
            id="modal-1",
            centered=True,
            is_open=False,
        ),
    ]
)

modal_Cluster = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader("エラー"),
                dbc.ModalBody("クラスタリングに失敗しました。"),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="close-2", className="ml-auto", n_clicks=0
                    )
                ),
            ],
            id="modal-2",
            centered=True,
            is_open=False,
        ),
    ]
)

@app.callback(
    Output('modal-set-item', 'is_open'),
    Output('select-item', 'data'),
    Input('set-item', 'n_clicks'),
    Input('close-set-item', 'n_clicks'),
    State('modal-set-item', 'is_open'),
    State('selected-item', 'value'),
    prevent_initial_call=True,
)
def show_set_item(n1, n2, is_open, checks):
    print('show_set_item')
    if n1 or n2:
        return not is_open, checks
    return is_open, checks

@app.callback(
     Output('select-item-list', 'children'),
    [Input('item-data', 'data')],
    prevent_initial_call=True,
)
def set_item_list(item):
    print('set_item_list')
    if item:
        item_checkbox = dcc.Checklist(
                            id='selected-item',
                            options=[{'label':f'{c}　', 'value':c} for c in item.keys()],
                            value=[],
                        )
    else:
        item_checkbox = list()
    return item_checkbox

@app.callback(
    [Output(f"collapse-{i}", "is_open") for i in range(1, 3)],
    Output("mols-data", "data"),
    Output("item-data", "data"),
    Output("data-table", 'children'),
    Output('modal-1', 'is_open'),
    Input('upload-data', 'contents'),
    Input('close-1', 'n_clicks'),
    State('upload-data', 'filename'),
    State('modal-1', 'is_open')
)
def upload_data(contents, n_close, filename, is_open):
    if contents is not None:
        try:
            if filename[-3:].upper() == "SDF":
                mols = Chem.SDMolSupplier(filename)
                df = PandasTools.LoadSDF(filename) # RDKitのメソッドを使用すること
                mols_data = [rdMolInterchange.MolToJSON(mol) for mol in mols]
            elif filename[-3:].upper() == "CSV":
                df = pd.read_csv(filename, header=0)
                in_smiles = [smiles for smiles in df.columns if 'SMILES' in smiles.upper()]
                mols = [Chem.MolFromSmiles(smiles) for smiles in df[in_smiles[0]]]
                mols_data = [rdMolInterchange.MolToJSON(mol) for mol in mols]
            df = df.dropna(how='all').dropna(how='all', axis=1)
            jsn = df.to_json(date_format='iso')
            item_data = eval(jsn)
        except:
            table = dbc.Table.from_dataframe(table_df("", "", ""), striped=True, bordered=True, hover=True)
            return True, False, dict(), dict(), table, not is_open

        len_record = len(df)
        len_column = len(df.columns)
        table = dbc.Table.from_dataframe(table_df(filename, len_record, len_column), striped=True, bordered=True, hover=True)
        return True, True, mols_data, item_data, table, False
    else:
        table = dbc.Table.from_dataframe(table_df("", "", ""), striped=True, bordered=True, hover=True)
        return True, False, dict(), dict(), table, False


@app.callback(
    Output("exe-btn", 'disabled'),
    [Input("cluster-arg", 'value'), 
     Input("cluster-num", 'value')],
)
def classify_data(select_arg, select_num):
    print('classify_data')
    if select_arg and select_num:
        return False
    return True 

@app.callback(
    Output({'type': 'result-num', 'index': ALL}, 'children'),
    Output({'type': "result-btn", 'index': ALL}, 'disabled'),
    Output({'type': "result-btn", 'index': ALL}, 'color'),
    Input('trigger', 'data'),
    State('cls-data', 'data'),
    State('modify-cls', 'data'), 
    State("cluster-num", 'value'),State("cls-data", 'data'),
    prevent_initial_call=True,
)
def update(trigger, cls_data, modify_cls, num, now_cls_data):
    print('update')
    time.sleep(SLEEP_TIME)
    if modify_cls:
        try:
            cls_cnt = [cls_data.count(i) for i in range(1, num+2)]
            btn_disabled = [False] * (num + 1)
        except:
            cls_cnt = [0] * (num + 1)
        if not cls_data:
            btn_disabled = [True] * (num + 1)
        color = ['danger'] * (num + 1)
    else:
        try:
            cls_cnt = [cls_data.count(i) for i in range(1, num+1)]
            btn_disabled = [False] * num
        except:
            cls_cnt = [0] * num
        if not cls_data:
            btn_disabled = [True] * num
        color = ['primary'] * num
        
    return [f"{i}" for i in cls_cnt], btn_disabled, color #, trigger_id

# @app.callback(
#     Output('debug', 'children'),
#     [Input('exe-btn', 'n_clicks'),
#      Input('close-2', 'n_clicks'),
#      Input("cluster-arg", 'value'), 
#      Input("cluster-num", 'value'),
#      Input("mols-data", "data"),
#      Input('modal-2', 'is_open'),],
#     prevent_initial_call=True,
# )
# def monitor(click, n_close, arg, num, mols_jsn, is_open):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         trigger_id = 'No clicks yet'
#     else:
#         trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
#     return  f'trigger:{trigger_id}'

@app.callback(
    Output('cls-data', 'data'),
    Output('modal-2', 'is_open'),
    Output('disp-mode', 'disabled'),
    Output('set-item', 'disabled'),
    Output('modify-cls', 'data'),
    # Output('debug', 'children'),
    Input('exe-btn', 'n_clicks'),
    Input('close-2', 'n_clicks'),
    Input("cluster-arg", 'value'), 
    Input("cluster-num", 'value'),
    Input({'type': "result-btn", 'index': ALL}, 'n_clicks'),
    Input({'type': 'check-card', 'index': ALL}, 'value'), 
    Input({'type': 'check-table', 'index': ALL}, 'value'), 
    State("mols-data", "data"),
    State('modal-2', 'is_open'),
    State('cls-data', 'data'),
    State('modify-cls', 'data'),
    prevent_initial_call=True,
)
def clustering(click, n_close, arg, num, result_btn, check_card, check_table, mols_jsn, is_open, now_cls_data, now_modify_cls):
    print('clustering')
    time.sleep(SLEEP_TIME)
    modify_cls = list()

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        now_dic = eval(trigger_id)
        now_type = now_dic['type']
        now_index = now_dic['index']
    except:
        now_type = trigger_id
        now_index = now_modify_cls
    debug = f'{now_type} {now_index} {modify_cls} {now_cls_data}'

    if click is None:
        return list(), False, True, True, modify_cls#, debug
    elif now_type in ['cluster-arg', 'cluster-num']:
        raise PreventUpdate
    elif now_type == 'exe-btn':
        try:
            mols = [rdMolInterchange.JSONToMols(jsn)[0] for jsn in mols_jsn]
            len_data = len(mols)
            # Morganフィンガープリントの生成と距離行列の計算
            morgan_fp = [AllChem.GetMorganFingerprintAsBitVect(x,2,2048) for x in mols]
            dis_matrix = [DataStructs.BulkTanimotoSimilarity(morgan_fp[i], morgan_fp[:len_data],returnDistance=True) for i in range(len_data)]
            dis_array = np.array(dis_matrix)
            # クラスタリングの実行
            clusters = AgglomerativeClustering(n_clusters=num, linkage=arg)
            clusters.fit(dis_array)
            sr = pd.value_counts(clusters.labels_)
            map_dic = {key:value for key, value in zip(sr.index, range(1, len(sr)+1))}
            cls_data = [map_dic[i] for i in clusters.labels_]
            return cls_data, False, False, False, now_modify_cls#, debug
        except:
            return list(), not is_open, True, True, modify_cls#, debug
    elif now_type in ['check-card', 'check-table']:
        modify_card = [i[0] for i in check_card if i!=[]]
        modify_table = [i[0] for i in check_table if i!=[]]
        if modify_card:
            modify_cls = modify_card
        elif modify_table:
            modify_cls = modify_table
        else:
            modify_cls = list()
        return now_cls_data, False, False, False, modify_cls#, debug

    elif now_type == 'result-btn':
        return now_cls_data, False, False, False, now_modify_cls#, debug

app.layout = dbc.Container([
                            title, 
                            debug_row,
                            accordion,
                            modal_set_item,
                            modal_load_data,
                            modal_Cluster,
                            dcc.Store(id='mols-data', data=list()),
                            dcc.Store(id='item-data', data=dict()),
                            dcc.Store(id='cls-data', data=list()),
                            dcc.Store(id='select-item', data=list()),
                            dcc.Store(id='select-cls', data=0),
                            dcc.Store(id='modify-cls', data=list()),
                            dcc.Store(id='trigger', data=0),
                            ])


app.run_server(host='0.0.0.0', port=8001, debug=True)