from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
from rdkit.Chem.Draw import rdMolDraw2D, IPythonConsole
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
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

def table_df(file_name, records, columns):
    df = pd.DataFrame(
        {
            "項　目": ["ファイル名", "レコード数", "項　目　数"],
            "内　　　容": [f"{file_name}", f"{records}", f"{columns}"],
        }
    )
    return df

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.YETI], # SUPERHERO BOOTSTRAP
                )

# View
title = html.Div([
    dbc.Row([
        dbc.Col(
            dbc.Alert(html.H3("Webアプリ　Chemoinfomaticsデモ"), 
                      color="primary", style={'text-align':'center'}
                      )
        ),
    ], justify="center")
])

upload_row = html.Div([
    dbc.Row([
        dbc.Col(
            dcc.Upload(
                id='upload-data',
                children=html.Div(
                    ['Drag and Drop SDF',
                    html.Br(), 'or', html.Br(),
                    html.A('Select Files')
                    ]),
                style={
                    'width': '80%',
                    'height': '100%',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                    },
                multiple=False,
            ), width=4
        ),
        dbc.Col(
            dcc.Loading(
                id='loading',
                type='default',
                children=[
                    dbc.Table(
                        # dbc.Table.from_dataframe(data_df("", "", ""), striped=True, bordered=True, hover=True), 
                        id="data-table",
                    )
                ]
            ), width=6
        ),
        dbc.Col(
            dbc.Button('参照・抽出', id="filter_btn", color="primary", disabled=True, className="btn btn-md"), width=2, style={'text-align':'center'}, align='start'
        ),
    ], no_gutters=False, justify="start")
], style={'margin':'10px 10px 0px 10px'}) # マージン：上　右　下　左

class_dic ={'ward':'ウォード法', 'average':'平均連結法', 'complete':'完全連結法'}

class_row = html.Div([
    dbc.Row([
        dbc.Col(
            html.H6('方法：'), width=1, style={'text-align':'right'}, align='end'
        ),
        dbc.Col(
            dcc.Dropdown(
                id='cluster-arg',
                options=[{'label': item, 'value': key} for key, item in class_dic.items()],
                clearable=False, 
                persistence=False, 
                persistence_type='local',
            ), width=2, align='end',
        ),
        dbc.Col(
            html.H6('分割数：'), width=1, style={'text-align':'right'}, align='end'
        ),
        dbc.Col(
            dcc.Dropdown(
                id='cluster-num',
                options=[{'label': item, 'value': item} for item in range(2,11)],
                clearable=False, 
                persistence=False, 
                persistence_type='local',
            ), width=1, align='end',
        ),
        dbc.Col(
            html.H6('ＳＭＩＬＥＳ：'), width=2, style={'text-align':'right'}, align='end'
        ),
        dbc.Col(
            dcc.Dropdown(
                id='cluster-sml',
                clearable=False, 
                persistence=False, 
                persistence_type='local',
            ), width=4, align='end',
        ),
        dbc.Col(
            dbc.Button('実 行', id="exe-btn", color="primary", disabled=True, className="btn btn-md"), width=1, style={'text-align':'center'}, align='end'
        ),
    ], no_gutters=True, justify="start"), 
], style={'margin':'10px 0px 10px 0px'}) # マージン：上　右　下　左

def result_col(i):
    return dbc.Col([
            dbc.Button([f"{i}", html.Br(),
                dbc.Badge(id={'type':'result-num', 'index':i}, pill=True, color="warning", className="ml-1")],
                color="primary",
                outline=True, className="btn btn-lg btn-block"
                ),
        ], width=1)

result_row = html.Div([
    dbc.Row(id="result-row", no_gutters=True, justify="center")
], style={'margin':'10px 10px 10px 10px'}) # マージン：上　右　下　左

@app.callback(
    Output("result-row", 'children'),
    [Input("cluster-arg", 'value'),
     Input("cluster-num", 'value'),
     Input("cluster-sml", 'value'),],
    prevent_initial_call=True,
)
def set_result(arg, num, sml):
    if num:
        row = [result_col(i) for i in range(1, num+1)]
    else:
        raise PreventUpdate
    return row


debug_row = html.Div([
    dbc.Row([
        dbc.Col(
            html.Label(id="debug"), width=12
        )
    ], style = {'display': 'block'}), 
    # ], style = {'display': 'none'}), 
]) 

row_dic = {1:upload_row, 2:class_row, 3:result_row}

def make_item(i):
    # we use this function to make the example items to avoid code duplication
    label_dic = {1:"データの読込", 2:"クラスタリング", 3:"結　　果"}
    return dbc.Card(
        [
            # dbc.Button(
            dbc.Alert(
                f"{label_dic[i]}",
                color="primary",
                id=f"group-{i}-toggle",
                # active=True,
                # n_clicks=0,
                style={'text-align':'left'},
            ),
            dbc.Collapse(
                row_dic[i],
                id=f"collapse-{i}",
                is_open=False,
            ),
        ]
    )

accordion = html.Div(
    [make_item(1), make_item(2), make_item(3)],
)

modal_load_data = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader("エラー"),
                dbc.ModalBody("ファイルの読込みに失敗しました。"),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="close_1", className="ml-auto", n_clicks=0
                    )
                ),
            ],
            id="modal_1",
            centered=True,
            is_open=False,
        ),
    ]
)

modal_SMILES = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader("エラー"),
                dbc.ModalBody("SMILESの解析に失敗しました。"),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="close_2", className="ml-auto", n_clicks=0
                    )
                ),
            ],
            id="modal_2",
            centered=True,
            is_open=False,
        ),
    ]
)

@app.callback(
    [Output(f"collapse-{i}", "is_open") for i in range(1, 3)],
     Output("data", "data"),
     Output("data-table", 'children'),
     Output("filter_btn", 'disabled'),
     Output("cluster-sml", 'options'),
     Output("cluster-sml", 'value'),
     Output('modal_1', 'is_open'),
    [Input('upload-data', 'contents'),
     Input('close_1', 'n_clicks')],
    [State('upload-data', 'filename'),
     State('modal_1', 'is_open')]
)

def upload_data(contents, n_close, filename, is_open):
    if contents is not None:
        try:
            if filename[-3:].upper() == "SDF":
                df = PandasTools.LoadSDF(filename) # RDKitのメソッドを使用すること
            elif filename[-3:].upper() == "CSV":
                df = pd.read_csv(filename, header=0)
            df = df.dropna(how='all').dropna(how='all', axis=1)
            jsn = df.to_json(date_format='iso')
            dic_data = eval(jsn)
        except:
            table = dbc.Table.from_dataframe(table_df("", "", ""), striped=True, bordered=True, hover=True)
            return True, False, dict(), table, True, [], None, not is_open

        len_record = len(df)
        len_column = len(df.columns)
        table = dbc.Table.from_dataframe(table_df(filename, len_record, len_column), striped=True, bordered=True, hover=True)
        in_smiles = [smiles for smiles in df.columns if 'SMILES' in smiles.upper()]
        no_smiles = [smiles for smiles in df.columns if 'SMILES' not in smiles.upper()]
        columns = in_smiles + no_smiles
        options=[{'label': column, 'value': column} for column in columns]

        return True, True, dic_data, table, False, options, None, False
    else:
        table = dbc.Table.from_dataframe(table_df("", "", ""), striped=True, bordered=True, hover=True)
        return True, False, dict(), table, True, [], None, False

@app.callback(
    Output("exe-btn", 'disabled'),
    [Input("cluster-arg", 'value'), 
     Input("cluster-num", 'value'),
     Input("cluster-sml", 'value')],
)
def classify_data(select_arg, select_num, select_sml):
    if select_arg and select_num and select_sml:
        return False
    return True 


@app.callback(
    Output(f"collapse-3", "is_open"),
    Output({'type': 'result-num', 'index': ALL}, 'children'),
    Output('modal_2', 'is_open'),
    [Input('exe-btn', 'n_clicks'),
     Input('close_2', 'n_clicks')],
    [State("cluster-arg", 'value'), 
     State("cluster-num", 'value'),
     State("cluster-sml", 'value'),
     State("data", "data"),
     State('modal_2', 'is_open'),],
    prevent_initial_call=True,
)
def upload_data(click, n_close, arg, num, sml, data, is_open):
    if click is None:
        return False, None, False
    else:
        try:
            len_sml = len(data[sml])
            mols = [Chem.MolFromSmiles(smiles) for smiles in data[sml].values()]
            # Morganフィンガープリントの生成と距離行列の計算
            morgan_fp = [AllChem.GetMorganFingerprintAsBitVect(x,2,2048) for x in mols]
            dis_matrix = [DataStructs.BulkTanimotoSimilarity(morgan_fp[i], morgan_fp[:len_sml],returnDistance=True) for i in range(len_sml)]
            dis_array = np.array(dis_matrix)
            # クラスタリングの実行
            clusters = AgglomerativeClustering(n_clusters=num, linkage=arg)
            clusters.fit(dis_array)
            sr = pd.value_counts(clusters.labels_)
            return True, [f"{i:>8}" for i in list(sr)], False
        except:
            return True, [[] for i in range(num)], not is_open

app.layout = dbc.Container([
                            title, 
                            accordion,
                            debug_row,
                            modal_load_data,
                            modal_SMILES,
                            dcc.Store(id='data'),
                            ])


app.run_server(host='0.0.0.0', port=8001, debug=True)