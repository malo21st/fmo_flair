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
# import plotly.graph_objects as go
# from flask_caching import Cache
# import uuid

MAX_BUTTON = 25
MAX_CLUSTERING = 15

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.YETI], # SUPERHERO BOOTSTRAP
                suppress_callback_exceptions=True,
                )

#v View ============================================================================================
##v Title
title = html.Div([
    dbc.Row([
        dbc.Col(
            dbc.Alert(html.H3("fmo-flair プロトタイプ"), 
                      color="primary", style={'text-align':'center'}
                      )
        ),
    ], justify="center")
])

##v Data Load & set param
class_dic ={'ward':'ウォード法', 'average':'平均連結法', 'complete':'完全連結法'}

def table_df(file_name, records, columns):
    df = pd.DataFrame(
        {
            "項　目": ["ファイル名", "レコード数", "項　目　数"],
            "内　　　容": [f"{file_name}", f"{records}", f"{columns}"],
        }
    )
    return df

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
                    options=[{'label': item, 'value': item} for item in range(2, MAX_CLUSTERING + 1)],
                    # value=2,
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


##v Result Data
result_row = html.Div([
    dbc.Row(
        [dbc.Col([
            dbc.Button([f"{i}", html.Br(),
                dbc.Badge("0", id={'type':'result-num', 'index':i}, pill=True, color="warning", className="ml-1")],
                id={'type':'result-btn', 'index':i},
                color="primary",
                outline=True,
                active=False,
                disabled=True,
                n_clicks=0,
                className="btn btn-lg btn-block",
                style={'display': 'none'}
                ),
        ], width=1) for i in range(1, MAX_BUTTON + 1)],
        id="result-row", no_gutters=True, justify="start"
    ),
    dbc.Row(id="show-mols", no_gutters=False, justify="start", style={'margin':'10px 0px 0px 0px'})

], style={'margin':'10px 10px 10px 10px'}) # マージン：上　右　下　左

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

##v アコーディオン
def make_item(i):
    row_dic = {1:class_row, 2:result_row}
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
        item = None
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


##v Dev & Debug
debug_row = html.Div([
    dbc.Row([
        dbc.Col(
            html.Label(id="debug"), width=12
        )
    ], style = {'display': 'block'}), 
    # ], style = {'display': 'none'}), 
]) 

##v Layout
app.layout = dbc.Container([
                            title, 
                            debug_row,
                            accordion,
                            modal_set_item,
                            modal_load_data,
                            modal_Cluster,
                            dcc.Store(id='mols-data', data=dict()), # 構造式のJsonリスト
                            dcc.Store(id='item-data', data=dict()), # 構造式以外の項目のJsonリスト
                            dcc.Store(id='cls-data', data=list()), # 分類結果のリスト
                            dcc.Store(id='select-item', data=list()), # 表示項目のリスト
                            dcc.Store(id='select-cls', data=0), # 選択した分類番号
                            dcc.Store(id='temp-data', data=dict()), # 一時的に一覧表示に使用する辞書
                            dcc.Store(id='modify-data', data=dict()), # 手動分類で変更するデータの辞書
                            dcc.Store(id='trigger', data=0),
                            ])


#c Controller & Model
##c Data Load データの読込み ===============================================================================
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

##c Result Data クラスタリング結果 =========================================================================
@app.callback(
    Output("exe-btn", 'disabled'),
    Input("cluster-arg", 'value'), 
    Input("cluster-num", 'value'),
    Input({'type': 'check-card', 'index': ALL}, 'value'), 
    Input({'type': 'check-table', 'index': ALL}, 'value'), 
)
def active_exe_btn(select_arg, select_num, check_card, check_table): # 実行ボタンの活性化
    if select_arg and select_num:
        return False

    modify_card = [i[0] for i in check_card if i!=[]]
    modify_table = [i[0] for i in check_table if i!=[]]
    if len(modify_card) + len(modify_table) > 0:
        return False
    return True 

@app.callback(
    Output('cls-data', 'data'),
    Output('modal-2', 'is_open'),
    Input('exe-btn', 'n_clicks'),
    Input('modify-data', 'data'),
    Input('close-2', 'n_clicks'),
    Input("cluster-arg", 'value'), 
    Input("cluster-num", 'value'),
    State("mols-data", "data"),
    State('modal-2', 'is_open'),
    State('cls-data', 'data'),
    prevent_initial_call=True,
)
def clustering(click, modify_data, n_close, arg, num, mols_jsn, is_open, old_cls_data): # クラスタリングの実施
    changed_id = dash.callback_context.triggered[0]['prop_id']

    if 'exe-btn' in changed_id:
        try:
            mols = [rdMolInterchange.JSONToMols(jsn)[0] for jsn in mols_jsn]
            len_data = len(mols)
            # Morganフィンガープリントの生成と距離行列の計算
            morgan_fp = [AllChem.GetMorganFingerprintAsBitVect(x,2,2048) for x in mols]
            dis_matrix = [DataStructs.BulkTanimotoSimilarity(morgan_fp[i], morgan_fp[:len_data],returnDistance=True) for i in range(len_data)]
            dis_array = np.array(dis_matrix)
            # クラスタリングの実行
            clusters = AgglomerativeClustering(n_clusters=num, linkage=arg) # num と arg をセット
            clusters.fit(dis_array)
            sr = pd.value_counts(clusters.labels_)
            map_dic = {key:value for key, value in zip(sr.index, range(1, len(sr)+1))}
            cls_data = [map_dic[lbl] for lbl in clusters.labels_]
            return cls_data, False
        except:
            return [1,1,1], not is_open
    elif 'modify-data' in changed_id:
        for key, value in modify_data.items():
            old_cls_data[int(key)] = value
        return old_cls_data, False
    elif 'cluster-arg' in changed_id or 'cluster-num' in changed_id:
        return list(), False
    else:
        return [2,2,2,2], not is_open
        # raise PreventUpdate


@app.callback(
    Output({'type': 'result-num', 'index': ALL}, 'children'),
    Output({'type': "result-btn", 'index': ALL}, 'disabled'),
    Output({'type': "result-btn", 'index': ALL}, 'color'),
    Output({'type': "result-btn", 'index': ALL}, 'active'),
    Output({'type': "result-btn", 'index': ALL}, 'style'),
    Output('disp-mode', 'disabled'),
    Output('set-item', 'disabled'),
    Input("cls-data", 'data'),
    Input({'type': 'check-card', 'index': ALL}, 'value'), 
    Input({'type': 'check-table', 'index': ALL}, 'value'), 
    State('cluster-num', 'value'),
    State('select-cls', 'data'),
    prevent_initial_call=True,
)
def toll_data(cls_data, check_card, check_table, cls_num, select_cls): # 分類結果の集計
    changed_id = dash.callback_context.triggered[0]['prop_id']
    try:
        num = max(cls_num, len(set(cls_data)))
        cls_cnt = [cls_data.count(i) for i in range(1, num+1)]
        btn_disabled = [False] * (MAX_BUTTON)
    except:
        cls_cnt = list()
        btn_disabled = [True] * (MAX_BUTTON)

    if not cls_data:
        btn_disabled = [True] * (MAX_BUTTON)
        btn_style = [{'display': 'none'}] * MAX_BUTTON
    else:
        btn_style = [{'display': 'block'}] * num + [{'display': 'none'}] * (MAX_BUTTON-num)

    btn_color = ['primary'] * (MAX_BUTTON)

    modify_card = [i[0] for i in check_card if i!=[]]
    modify_table = [i[0] for i in check_table if i!=[]]

    if 'check-card' in changed_id or 'check-table' in changed_id:
        btn_num = num + 1
        if len(modify_card) + len(modify_table) > 0:
            cls_cnt += [0]
            cls_cnt = [cls_data.count(i) for i in range(1, num+2)]
            btn_disabled = [False] * (MAX_BUTTON)
            btn_color = ['danger'] * (MAX_BUTTON)
            btn_num += 1
            btn_style = [{'display': 'block'}] * (num+1) + [{'display': 'none'}] * (MAX_BUTTON-num-1)
        else:
            cls_cnt = [cls_data.count(i) for i in range(1, num+1)]
            cls_cnt += [0]
            btn_disabled = [False] * (MAX_BUTTON)
            btn_color = ['primary'] * (MAX_BUTTON)
            btn_style = [{'display': 'block'}] * num + [{'display': 'none'}] * (MAX_BUTTON-num)

    result_num = [f"{i}" for i in cls_cnt] + [0] * (MAX_BUTTON - len(cls_cnt))

    btn_active = [False] * (MAX_BUTTON) #btn_num
    try:
        btn_active[select_cls - 1] = True
    except:
        pass
        # btn_active = dash.no_update

    return result_num, btn_disabled, btn_color, btn_active, btn_style, False, False


##c Detail Data 詳細なデータ ==========================================================================
def mol_img(num, width, height, mols_data):
    try:
        mols = [rdMolInterchange.JSONToMols(jsn) for jsn in mols_data]
        mol = mols[int(num)][0]
        mol.RemoveAllConformers()
        rdCoordGen.AddCoords(mol)
        img = Draw.MolToImage(mol, size=(width,height))
    except Exception as error:
        img = "Error"
        print(error)
    return img

def card(num, cls, mols_data, item_data, select_item):
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
                html.P(f"{item_data[item][str(num)]}", className="card-text")]) for item in select_item],
            ),
    ]
    return card_content

@app.callback(
    Output('select-item-list', 'children'),
    Input('item-data', 'data'),
    prevent_initial_call=True,
)
def set_item_list(item):
    if item:
        item_checkbox = dcc.Checklist(
                            id='selected-item',
                            options=[{'label':f'{c}　', 'value':c} for c in item.keys()],
                            value=[],
                        )
    else:
        item_checkbox = None
    return item_checkbox


@app.callback(
    Output('temp-data', 'data'),
    Output('modify-data', 'data'),
    Output('select-cls', 'data'),
    Output({'type': 'check-card', 'index': ALL}, 'value'), 
    Output({'type': 'check-table', 'index': ALL}, 'value'), 
    Input({'type': "result-btn", 'index': ALL}, 'n_clicks'),
    Input('cluster-num', 'value'),
    Input('cluster-arg', 'value'),
    State('cls-data', 'data'),
    State({'type': 'check-card', 'index': ALL}, 'value'), 
    State({'type': 'check-table', 'index': ALL}, 'value'), 
    State('select-cls', 'data'),
    prevent_initial_call=True,
)
def classification_list(_1, _2, _3, cls_data, check_card, check_table, selected_cls):
    # if dash.callback_context.triggered[0]['value'] == 0:
    #     raise PreventUpdate

    trigger_id = dash.callback_context.triggered[0]['prop_id']

    if 'cluster-num' in trigger_id or 'cluster-arg' in trigger_id:
        return dict(), dict(), 12, check_card, check_table
    elif 'result-btn' in trigger_id:
        clicked_id_dic = eval(trigger_id.split('.')[0])
        select_cls = clicked_id_dic['index']

        modify_card_lst = [int(i[0]) for i in check_card if i!=[]]
        modify_table_lst = [int(i[0]) for i in check_table if i!=[]]
        modify_lst = modify_card_lst if modify_card_lst else modify_table_lst
        print(modify_lst)

        if not modify_lst: # 手修正がない場合
            extra_data = {i:select_cls for i in range(len(cls_data)) if cls_data[i]==select_cls}
            modify_data = dict()
            print(f'not modify:{extra_data}')
            return extra_data, modify_data, select_cls, check_card, check_table
        else: # 手修正がある場合
            extra_data = {i:selected_cls for i in range(len(cls_data)) if cls_data[i]==selected_cls}
            for key in modify_lst:
                extra_data[key] = select_cls
            modify_data = {key:select_cls for key in modify_lst}
            print(f'modify:{extra_data}')
            return extra_data, modify_data, selected_cls, check_card, check_table

@app.callback(
    Output("show-mols", 'children'),
    Input('temp-data', 'data'),
    Input('disp-mode', 'value'),
    Input('select-item', 'data'),
    State('mols-data', 'data'),
    State('item-data', 'data'),
    State('select-cls', 'data'),
    prevent_initial_call=True,
)
def show_mols(temp_data, disp_mode, select_item, mols_data, item_data, select_cls):

    if temp_data:
        if disp_mode == 'CARD':
            mols_cnt = [dbc.Col([
                        dbc.Card(
                            card(num, cls_num, mols_data, item_data, select_item), 
                            color="light", inverse=False,
                            style={'max-width':'100%'}
                            ),
                        html.P(),
                        ], lg=2, md=3, sm=4, xs=4
                    ) for num, cls_num in temp_data.items()]
        elif disp_mode == 'TABLE':
            if select_cls:
                df = pd.DataFrame({
                        '選択':[dcc.Checklist(
                                id={'type':'check-table', 'index':num}, 
                                options=[{'label':'', 'value':num}],
                                value=[]) for num in temp_data.keys()],
                        '区分':temp_data.values(),
                        'No.':temp_data.keys(),
                        '構造式':[html.Img(src=mol_img(m, 100, 100, mols_data)) for m in temp_data.keys()],
                    })
                for item in select_item:
                    df[item]=[item_data[item][str(num)] for num in temp_data.keys()]
                mols_cnt = [dbc.Col(
                                dbc.Table.from_dataframe(
                                    df, striped=False, bordered=True, hover=True,
                                    responsive=True, className="card-text",
                                ),
                            width=12)]
            else:
                mols_cnt = None
    else:
        mols_cnt = ""
        
    return mols_cnt

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
    if n1 or n2:
        return not is_open, checks
    return is_open, checks


##c Dev & Debug ======================================================================================
@app.callback(
    Output('debug', 'children'),
    Input('exe-btn', 'n_clicks'),
    Input({'type': "result-btn", 'index': ALL}, 'n_clicks'),
    Input('close-2', 'n_clicks'),
    Input("cluster-arg", 'value'), 
    Input("cluster-num", 'value'),
    Input('disp-mode', 'value'),
    Input('set-item', 'n_clicks'),
    Input("mols-data", "data"),
    Input('cls-data', 'data'),
    Input('modal-2', 'is_open'),
    prevent_initial_call=True,
)
def monitor(*var):
    trigger_id = dash.callback_context.triggered[0]['prop_id']
    value = dash.callback_context.triggered[0]['value']
    return  f'trigger：{trigger_id}　　value：{str(value)[:130]}'



app.run_server(host='0.0.0.0', port=8001, debug=True)