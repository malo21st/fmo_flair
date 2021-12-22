import dash 
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash_table.Format import Format, Scheme, Trim
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly import subplots
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import base64
import datetime
import io
import json

from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools, rdMolInterchange, rdCoordGen
from rdkit.Chem.Draw import rdMolDraw2D, IPythonConsole


INFO_COLS = [' # ', 'Column', 'Non-Null Count', 'Dtype']
PARAMS = ['RES_1', 'RES_2', 'DIST', 'TOTAL', 'ES', 'EX', 'CT', 'DI']
COLS_OPTION = [{"label": "", "value": ""}]

df = pd.DataFrame()
df_fmo = pd.DataFrame()
df_smiles = pd.read_pickle('fmodbid_simles.pkl')

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP], # SUPERHERO BOOTSTRAP YETI 
                meta_tags=[{
                    'name': 'viewport',
                    'content': 'width=device-width, initial-scale=1.0'
                    }]
                )

# View
title_row = html.Div([
    dbc.Row([
        dbc.Col(
            dbc.Alert(html.H3("FMO計算結果一覧デモ"), 
                      color="primary", style={'text-align':'center'}
                      )
        ),
    ], justify="center")
])


first_row = html.Div([
    dbc.Row([
        dbc.Col(
            dcc.Upload(
                id='id-upload-files',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=True,
            ), width=3
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.P(
                        id='id-files-info',
                        className="card-text",
                    ),
                ]),
            ],), width=9
        )
    ], justify="center", align="center")
])

second_row = html.Div([
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dash_table.DataTable(
                    id='id-columns-info',
                    editable=True,
                    columns=[{"name":'Param', "id":'Param', 'presentation':'dropdown'}] + [{"name":i, "id":i, 'editable':False} for i in INFO_COLS],
                    dropdown={'Param': {'options': [{'label': prm, 'value': prm} for prm in PARAMS]}},
                    style_header={'text-align':'left'},
                    style_cell={'text-align':'left'},
                    style_cell_conditional=[
                        {'if': {'column_id': INFO_COLS[1]},
                         'editable': False},
                        {'if': {'column_id': 'Param'},
                         'background-color': 'lightyellow', 'width': '100px'},
                    ],                    
                    css=[{"selector": ".Select-menu-outer", "rule": "display: block !important"}],
                ),
            ],), width=12
        )
    ], justify="center")
])

third_row = html.Div([
    html.Br(),
    dbc.Row([
        dbc.Col(
            dbc.Button('処　理', id="id-btn-process", color="primary", className="mr-1"),
            width=2
        )
    ], justify="start")
])

forth_row = html.Div([
    html.Br(),
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dcc.Graph(id="id-result-2axis"),
                dcc.Graph(id="id-result-scatter"),
            ],), id="id-result-2graph", width=12, style = {'display': 'none'},
        )
    ], justify="left")
])

IFIE_6graph = html.Div([
    dbc.Modal([
        dbc.Row([
            dbc.Col([
                dbc.ModalBody([
                    html.H3(id='id-data-name'),
                    dbc.InputGroup([
                        dbc.InputGroupAddon("DISTの閾値", addon_type="prepend"), 
                        dbc.Input(id='id-threshold', debounce=True, value='5'),
                        dbc.Button('グラフ', id='id-result-6graph'),
                        html.Label('　　'),
                        dbc.Button("Close", id="close-6graph", className="ml-auto", n_clicks=0),
                    ], style={'margin':'30px 0px 30px 0px'}), # マージン：上　右　下　左
                ]),
            ], width=5),
            dbc.Col([
                dbc.CardImg(id='id-mol-img', top=True, bottom=False),
            ], width=3),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='id-6graph'),
            ]),
        ], id="id-show-6graph", style={'width':'100%', 'display': 'none'}),
    ],
    id="modal-6graph",
    size="xl",
    is_open=False,
    ),
])


debug_row = html.Div([
    dbc.Row([
        dbc.Col(
            html.Div(id='id-debug'),
            width = 12,
        ),
    ], justify="center")
])

app.layout = dbc.Container([
                            title_row,
                            first_row, 
                            second_row,
                            third_row,
                            forth_row,
                            IFIE_6graph,
                            dcc.Store(id='id-data'),
                            debug_row,
                            ])

def df_to_info_table(df):
    buf = io.StringIO()
    df.info(buf=buf)
    buf_val = buf.getvalue().split('\n')
    info = buf_val[3:-3]

    start = [info[0].find(w) for w in INFO_COLS] + [len(info[0])]
    rows = [[info[k][i:j].strip() for i,j in zip(start[:-1], start[1:])] for k in range(2, len(info)-1)]
    data = [{i:r for i, r in zip(INFO_COLS, row)} for row in rows]
    return data

@app.callback(
    Output('id-files-info', 'children'),
    Output("id-columns-info", 'data'),
    Input('id-upload-files', 'contents'),
    State('id-upload-files', 'filename'),
    PreventUpdate = True,
)
def upload_data(list_of_contents, list_of_names):
    global df
    if list_of_contents is not None:
        for contents, filename in zip(list_of_contents, list_of_names):
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            try:
                if 'csv' in filename:
                    temp_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0)
                    temp_df['name'] = filename[:-4]
                    df = pd.concat([df, temp_df])
            except Exception as e:
                print(e)
                return f'There was an error processing {filename}.', [html.P('ERROR')]

        info_table = df_to_info_table(df)

        return f"{len(list_of_names)}個のファイルを読込みました。", info_table
    else:
        msg = {col:value for col, value in zip(INFO_COLS, ['','ファイルを読み込むとカラム情報が表示されます。','',''])}
        return 'ファイルを読み込んで下さい。', [msg]

def data_to_graph(param):
    global df_fmo

    def res_num(row):
        return str(row.iloc[0]) + str(row.iloc[1])

    df_res = df[[param['RES_1'], param['RES_2']]]
    df_res.loc[:, ('RES')] = df_res.apply(res_num, axis=1)

    df_ifie = df[[param['DIST'], param['TOTAL'], param['ES'], param['EX'], param['CT'], param['DI']]]
    df_fmo = pd.concat([df[['name']], df_res[['RES']], df_ifie], axis=1)
    df_fmo.columns = ['NAME', 'RES','DIST','TOTAL IFIE', 'ES', 'EX', 'CT+mix', 'DI']
    df_sum = df_fmo.groupby('NAME', as_index=False).sum()
    df_sum_sort = df_sum.sort_values(by='TOTAL IFIE')

    df_ifie_sum = df_sum_sort[['NAME', 'TOTAL IFIE']]
    df_ifie_sum.columns = ['NAME', 'IFIE SUM']

    df_di_sum = df_sum_sort[['NAME', 'DI']]
    df_di_sum.columns = ['NAME', 'DI SUM']

    # df_di_sum = df_di_sum.sample(len(df_di_sum)//3)
    df_di_sum = pd.merge(df_di_sum, df_smiles, left_on='NAME', right_on='fmodbid').dropna(how='any')
    df_scatter = pd.concat([df_di_sum, df_ifie_sum[['IFIE SUM']]], axis=1).dropna(how='any')

    df_scatter = df_scatter.sort_values('Median(IC50)')

    X = df_scatter[['Median(IC50)']].values
    Y = df_scatter[['IFIE SUM']].values
    lr = LinearRegression()
    lr.fit(X, Y) 

    # fig 0
    trace0_1 = go.Bar(x=df_ifie_sum['NAME'], y=df_ifie_sum['IFIE SUM'], yaxis='y1', name='IFIE SUM')
    trace0_2 = go.Scatter(x=df_di_sum['NAME'], y=df_di_sum['DI SUM'], yaxis='y2', name='DI SUM',
                          mode='markers', marker_symbol='hexagram', marker_size=10)
    layout0 = go.Layout(
        xaxis = dict(title = 'NAME'),
        yaxis = dict(side = 'left', title='IFIE SUM [kcal/mol]'),
        yaxis2 = dict(side = 'right', overlaying = 'y', title='DI SUM [kcal/mol]'),
        )
    fig0 = go.Figure(data = [trace0_1, trace0_2], layout = layout0)

    # fig 1
    trace1_1 = go.Scatter(x=df_scatter['Median(IC50)'], y=df_scatter['IFIE SUM'], 
        text=df_scatter['NAME'], mode='markers', marker_size=10, showlegend=False)
    trace1_2 = go.Scatter(x=df_scatter['Median(IC50)'], y=[x[0] for x in lr.predict(X)], 
        mode='lines', line=dict(color='red', width=2), name="OLS", text=["OLS"]*len(X))

    layout1 = go.Layout(
        xaxis = dict(title = 'Median(IC50)', type='log'),
        yaxis = dict(title= 'IFIE SUM [kcal/mol]'),
        )
    fig1 = go.Figure(data = [trace1_1, trace1_2], layout = layout1)
    fig1

    return fig0, fig1


@app.callback(
    Output("id-result-2axis", 'figure'),
    Output("id-result-scatter", 'figure'),
    Output("id-result-2graph", 'style'),
    Input('id-btn-process', 'n_clicks'),
    State('id-columns-info', 'data'),
    PreventUpdate = True,
)
def data_processing(n_clicks, data):
    if n_clicks is None:
        return dict(), dict(), {'display': 'none'}
    else:
        param_dic = dict()
        for cell in data:
            try:
                cell['Param']
                param_dic[cell['Param']] = cell['Column']
            except:
                pass
        fig0, fig1 = data_to_graph(param_dic)
        return fig0, fig1, {'display': 'block'}


@app.callback(
    Output('id-show-6graph', 'style'),
    Input('id-result-6graph', 'n_clicks'),
    Input('id-threshold', 'n_submit'),
    Input("close-6graph", "n_clicks"),
)
def is_6graph(click_show, submit_show, click_hide):
    changed_id = dash.callback_context.triggered[0]['prop_id']
    if "id-result-6graph" in changed_id:
        return {'display': 'block'}
    elif "id-threshold" in changed_id:
        return {'display': 'block'}
    elif "close-6graph" in changed_id:
        return {'display': 'none'}
    else:
        return {'display': 'none'}


def smiles_to_img(name):
    width=300
    height=300
    try:
        smiles = df_smiles[df_smiles['fmodbid']==name]["CANONICAL_SMILES"].values[0]
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol, size=(width,height))
    except Exception as error:
        img = ""
        print(f"ERROR:{error}")
    return img

@app.callback(
    Output("modal-6graph", "is_open"),
    Output("id-data-name", "children"),
    Output("id-data", "data"),
    Output("id-mol-img", "src"),
    Input("id-result-2axis", "clickData"),
    Input("id-result-scatter", "clickData"),
    Input("close-6graph", "n_clicks"),
    State("modal-6graph", "is_open"),
    PreventUpdate = True,
)
def toggle_modal(clickData1, clickData2, n2, is_open):
    changed_id = dash.callback_context.triggered[0]['prop_id']
    if "id-result-2axis" in changed_id:
        name = clickData1['points'][0]['x']
        img = smiles_to_img(name)
        return not is_open, name, name, img
    elif "id-result-scatter" in changed_id:
        name = clickData2['points'][0]['text']
        img = smiles_to_img(name)
        if name != "OLS":
            return not is_open, name, name, img
        else:
            return False, "", "", ""
    elif "close-6graph" in changed_id:
        return not is_open, "", "", ""
    else:
        return False, "", "", ""

PIEDA = {
    "TOTAL IFIE":{"+":"#0000ff", "-":"#ff0000"},
    "ES":{"+":"#0000ff", "-":"#ff0000"},
    "EX":{"+":"#ff00ff", "-":"#ffffff"},
    "CT+mix":{"+":"#ffffff", "-":"#00ffff"},
    "DI":{"+":"#ffffff", "-":"#00ff00"},
    "DIST":{"+":"#6f6f6f", "-":"#ffffff"},
} 

@app.callback(
    Output('id-6graph', 'figure'),
    Input('id-result-6graph', 'n_clicks'),
    Input('id-threshold', 'n_submit'),
    State('id-data', 'data'),
    State('id-threshold', 'value'),
    PreventUpdate = True,
)
def madal_6graph(n_clicks, n_submit, data_name, value):
    try:
        df_temp = df_fmo[df_fmo['NAME']==data_name]
    except:
        return dict()
    threshold = float(value)
    df_fmo_th = df_temp[df_temp['DIST'] <= threshold]
    df_fmo_th_sort = df_fmo_th.sort_values(by='DIST')

    items = ['TOTAL IFIE', 'DIST', 'ES', 'EX', 'CT+mix', 'DI']
    units = ['kcal/mol', 'Å'] + ['kcal/mol'] * 4
    titles = [f"{item} [{unit}]" for item, unit in zip(items, units)]

    fig = subplots.make_subplots(rows=3, cols=2, subplot_titles=titles)

    def get_trace(fig, item, row, col):
        def value_color(value, item):
            if value >= 0:
                return PIEDA[item]["+"]
            else:
                return PIEDA[item]["-"]

        x = df_fmo_th_sort['RES']
        y = df_fmo_th_sort[item]
        max_value = max([abs(max(y)), abs(min(y))])
        my_colors = [value_color(v, item) for v in y]
        fig.append_trace(go.Bar(x=x, y=y, marker_color=my_colors), row, col)
        return fig

    for i, item in enumerate(items):
        r = i // 2 + 1
        c = i % 2 + 1
        fig = get_trace(fig, item, r, c)
    fig['layout'].update(height=810, width=1100, showlegend=False)
    return fig


# dev & debug ============================
@app.callback(
    Output('id-debug', 'children'),
    Input('id-result-scatter', 'clickData'),
    PreventUpdate = True,
)
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)    
    # try:
    #     changed_id = dash.callback_context.triggered[0]
    # except:
    #     changed_id = {'prop_id':'ERROR', 'value':'ERROR'}
    # return [html.P(f"prop_id : {changed_id['prop_id']}　　value：{changed_id['value']}"),
    #         html.P(f"data : {data}")]

app.run_server(host='0.0.0.0', port=8001, debug=True)