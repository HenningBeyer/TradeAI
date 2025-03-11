""" home.py
 - main menu definition
 - choice between TSF and RLT
 - extra functionality of experiment loading and comparing
"""

import dash
import os
from dash import html, dcc, callback, callback_context
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import time
from experatai_frontend.utils import update_dict_wrapper
from dash.exceptions import PreventUpdate

dash.register_page(__name__, path="/home")

#Buttons
button_style = {"text-transform": "none",
        'white-space': 'nowrap',
        'margin-bottom' : '15px',
        'width' : '250px'}

extra_button_style = {"text-transform": "none",
        'white-space': 'nowrap',
        'margin-bottom' : '15px'}

button_start_exp = dbc.Button(
                "Start Experiment",
                href='/dataset-config/dataset-config',
                id='button_start_exp',
                color='secondary',
                style=button_style)

button_orderbook = dbc.Button(
                "Orderbook Datasets",
                href='/dataset-config/dataset-config', # WIP
                id='button_orderbook',
                color='secondary',
                style=button_style)

button_load_experiments= dbc.Button(
                "Load Experiment",
                id='button_load_experiments',
                color='secondary',
                style=extra_button_style)

button_compare_methods = dbc.Button(
                "Compare Models",
                id='button_compare_methods',
                color='secondary',
                style=extra_button_style)

#Main Menu Card
main_menu_card = dbc.Card([
        dbc.CardHeader(html.H2("Start Experimenting")),
        dbc.CardBody([
            dbc.Row([
                dbc.Stack([
                    dbc.Col([
                        #downsample_freq_dropdown
                        dbc.DropdownMenu(
                        label="Experiment Task",
                        id='task_dropdown',
                        children=[
                            dbc.DropdownMenuItem("TSF", id='dropdown_item_tsf'),
                            dbc.DropdownMenuItem("RLT", id='dropdown_item_rlt'),
                        ]), 
                    ]),            
                    dbc.Col([
                        html.Div([
                            html.Span("|ㅤ", style={'font-size' : '110%'}),
                            html.Span("TSF", style={'font-size' : '110%'},
                                className="small-text", id='task_span')
                            ]),
                        ])
                    ], direction="horizontal")
                ], align='center', justify='start'),
            html.Hr(),
            dbc.Row([
                dbc.Stack([
                    dbc.Col([
                        #downsample_freq_dropdown
                        dbc.DropdownMenu(
                        label="Dataset",
                        id='dataset_dropdown',
                        children=[
                            dbc.DropdownMenuItem("OHLCV",     id='dropdown_item_ohlcv'),
                            dbc.DropdownMenuItem("Orderbook", id='dropdown_item_orderbook'),
                        ]), 
                    ]),            
                    dbc.Col([
                        html.Div([
                            html.Span("|ㅤ",   style={'font-size' : '110%'}),
                            html.Span("OHLCV", style={'font-size' : '110%'},
                                className="small-text", id='dataset_span')
                            ]),
                        ])
                    ], direction="horizontal")
                ], align='center', justify='start'),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dcc.Loading([
                        button_start_exp], 
                        fullscreen=True,  
                        type="circle"),
                    ]),
                dbc.Col([
                    html.Div('(min - h)', style={'margin-bottom' : '15px','font-size' : '90%', 'color' : 'grey'}, className="small-text")
                    ])
                ], align='center'),
            dbc.Row([
                dbc.Col([
                    button_orderbook]),
                dbc.Col([html.Div('(ms - s)', style={'margin-bottom' : '15px','font-size' : '90%', 'color' : 'grey'},
                        className="small-text")])
                ], align='center'),
            dbc.Row([
                dbc.Col([html.Div('(WIP)', style={'margin-bottom' : '15px','font-size' : '90%', 'color' : 'grey'},
                        className="small-text")])
                ], align='center'),
            ])
        ])


extra_menu_card = dbc.Card([
        dbc.CardHeader(html.H2("Extras")),
        dbc.CardBody([
            dbc.Row(dbc.Col([button_load_experiments]), align='center'),
            dbc.Row(dbc.Col([button_compare_methods]), align='center'),
        ])
    ])

layout = dbc.Row([
            dbc.Col(main_menu_card, width="auto"), 
            dbc.Col(extra_menu_card, width="auto"),
        ], justify="center"),

task_lookup = {'dropdown_item_tsf' : 'TSF',
               'dropdown_item_rlt' : 'RLT'}

dataset_type_lookup = {'dropdown_item_ohlcv'     : 'OHLCV',
                       'dropdown_item_orderbook' : 'Orderbook'}

@callback(
     Output("task_span", "children"),
    [Input('dropdown_item_tsf', 'n_clicks'),
     Input('dropdown_item_rlt', 'n_clicks')],
     prevent_initial_call=True)
def update_task_from_dropdown(n1, n2):
    sel_id = callback_context.triggered_id
    task = task_lookup[sel_id]
    return task

@callback(
     Output("dataset_span", "children"),
    [Input('dropdown_item_ohlcv'    , 'n_clicks'),
     Input('dropdown_item_orderbook', 'n_clicks')],
     prevent_initial_call=True)
def update_dataset_from_dropdown(n1, n2):
    sel_id = callback_context.triggered_id 
    dataset_type = dataset_type_lookup[sel_id]
    return dataset_type

@callback(
     Output('experiment_info_storage', 'data'),
    [Input('button_start_exp'        , 'n_clicks')],
    [State('experiment_info_storage' , 'data'),,
     State('task_span'               , 'children')
     State('dataset_span'            , 'children')]
     prevent_initial_call=True)
def update_params_from_buttons(n1, info, task, ds_type):
    sel_id = callback_context.triggered_id
    new_info = {'experiment_task_name' : task,
                'dataset_type'         : ds_type}
    updated_params = update_dict_wrapper(info, new_info)
    return updated_params




## TODO: add hardware monitoring?? --> yes easy monitoring when doing different tasks!
    # --> add slide ribbon or similar --> 3 colored radio buttons (inactive while not used!)
    # 'cpu_usage'              : None,  # str(np.round(np.sum([gpu.memoryUsed for gpu in gpus])/1024,1)) + ' GB'
    # 'ram_usage'              : None, # str(np.round(psutil.virtual_memory().used*10**(-9),1)) + ' GB'
    # 'gpu_usage'              : None,  # str(np.round(np.sum([gpu.memoryUsed for gpu in gpus])/1024,1)) + ' GB'
    #'peak_RAM_usage'  : None,  # str(psutil.virtual_memory().percent) +' %'