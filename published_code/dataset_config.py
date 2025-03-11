from experatai_backend.data_loader import Crypto_LTSF_Dataset, Crypto_RL_Dataset
from experatai_backend.experiment import Experiment_Config
from experatai_frontend.utils import get_param_by_name, load_cached_params, get_path_by_path_name, update_dict_wrapper, cache_params
import dash
import json
import time
from dash import dcc, callback, dash_table, html
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash_bootstrap_templates import ThemeSwitchAIO
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime, date
from dash.exceptions import PreventUpdate
import numpy as np

dash.register_page(__name__)

"""
   dataset_config.py
   
   provides the frontend for dataset configurations.
   
   - depending on the task and dataset, certain elements will be disabled or invisible.
   - functions: dataset configuration, data visualization, feature engineering, ...
"""


local_params = load_cached_params()
cached_feature_groups    = get_param_by_name(local_params, 'feature_groups')
cached_extra_indicators  = get_param_by_name(local_params, 'extra_indicators')
cached_worldquant_groups = get_param_by_name(local_params, 'worldquant_groups')
cached_filtered_features = get_param_by_name(local_params, 'filtered_features')
cached_train_perc        = get_param_by_name(local_params, 'train_perc')
cached_val_perc          = get_param_by_name(local_params, 'val_perc')
cached_test_perc         = get_param_by_name(local_params, 'test_perc')
del local_params

pd_freq_mapping = {                              
                "1 min"  :   '1T', 
                "2 min"  :   '2T', 
                "5 min"  :   '5T', 
                "10 min" :   '10T',
                "15 min" :   '15T',
                "30 min" :   '30T',
                "1 h"    :   '1H', 
                "2 h"    :   '2H', 
                "4 h"    :   '4H', 
                "12 h"   :   '12H',
                "1 d"    :   '1D', 
            }

# new default values if undefined:
cached_train_perc        = 0.7 if cached_train_perc == None else cached_train_perc
cached_val_perc          = 0.1 if cached_val_perc   == None else cached_val_perc
cached_test_perc         = 0.2 if cached_test_perc  == None else cached_test_perc

button_style = {"text-transform": "none",
        'white-space': 'nowrap',
        'margin-bottom' : '15px'}

button_select_dataset = dcc.Upload(
        dbc.Button("Select Dataset",
            style= {"text-transform": "none",
                'white-space': 'nowrap',
                'margin-bottom' : '15px',
                'width' : '155px'
                }
                ),
        id='button_select_dataset',
        multiple=False,
        
    )

button_fetch_new_dataset = dbc.Button(
        "Fetch new Dataset",
        id='button_fetch_new_dataset',
        color='secondary',
        style= {"text-transform": "none",
                'white-space': 'nowrap',
                'margin-bottom' : '12px',
                'width' : '155px'
                },
    )

button_plot_selected_features = dbc.Button(
        "Plot Selected Features",
        id='button_plot_selected_features',
        color='secondary',
        style={"text-transform": "none",
        'white-space': 'nowrap',
        #'align'         : 'right',
        'margin-bottom' : '15px',
        'font-size'     : '90%',
        'margin-left'   : '64%',
        'margin-top'    : '-56px'
        },
    ) 

button_feature_engineering = dbc.Button(
        "Calculate Features",
        id='button_feature_engineering',
        color='secondary',
        style={ 'text-transform': 'none',
                'white-space'   : 'nowrap',
                'margin-left'   : '1.2%',
                #'margin-right'  : '2.5%',
                'margin-top'    : '10px',
                'margin-bottom' : '4px'},
    )

button_set_lookback_windows = dbc.Button(
        "Filter Features",
        id='button_set_lookback_windows',
        color='secondary',
        style={ 'text-transform': 'none',
                'white-space'   : 'nowrap',
                #'margin-left'   : '1.2%',
                #'margin-right'  : '2.5%',
                'margin-top'    : '10px',
                'margin-bottom' : '4px'},
    )

button_set_lookback_windows = dbc.Button(
        "Set Lookback Windows",
        id='button_set_lookback_windows',
        color='secondary',
        style={ 'text-transform': 'none',
                'white-space'   : 'nowrap',
                #'margin-left'   : '1.2%',
                #'margin-right'  : '2.5%',
                'margin-top'    : '10px',
                'margin-bottom' : '4px'},
    )

button_download_dataset = dbc.Button(
        "Download Dataset",
        id='button_download_dataset',
        disabled=True,
        color='secondary',
        style={ 'text-transform': 'none',
                'white-space'   : 'nowrap',
                'margin-bottom' : '5px',
                'width'         : '100%'},
    )

button_use_dataset = dbc.Button(
        "Use Dataset",
        href=None,
        disabled=True,
        id='button_use_dataset',
        color='primary',
        style={ 'text-transform': 'none',
                'white-space'   : 'nowrap',
                'margin-bottom' : '5px',
                'width'         : '100%'},
    )

#temp_df = pd.DataFrame({'A':[1,2,3,4,54,6]})
DATA_DIV_WIDTH = '100%'
temp_data = {' '*i : [' ' for i in range(10)] for i in range(5)}  # 5 cols with 10 empty datapoints
df = pd.DataFrame(temp_data)#pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')
for i in range(4):
    df = df.append(df)
data_table =  html.Div([
                dash_table.DataTable(
                    id='data_table',
                    data=df.to_dict('records'),
                    #columns=[dict(type='numeric', format=Format(precision=5, scheme=Scheme.fixed))],
                    columns=[{"name": i, "id": i, 'selectable' : False} for i in df.columns],
                    #tooltip_header={i: i for i in df.columns},
                    fixed_rows={'headers': True},
                    style_table={'height': '250px', 'overflowY': 'auto', 'overflowX': 'auto'},
                    style_cell={'minWidth': '100px', 'maxWidth': '100px', 'width': '100px',
                                'overflow': 'ellipsis', 'textOverflow': 'ellipsis', 
                                'text-align' : 'center', 'border'  : '1px grey'},
                    column_selectable='multi',
                    #row_selectable='multiple',
                    #selected_columns=[], 
                    page_current=0,
                    page_action='native',
                    page_size=20),
                button_plot_selected_features,
                    
                ], style={'border'  : '1px white',
                          'width' : DATA_DIV_WIDTH,
                          'margin-bottom' : '15px'})

feature_graph = html.Div([
                dcc.Graph(figure=go.Figure(), id='feature_graph')
                ], style={'width' : DATA_DIV_WIDTH})



indicator_dropdown  =  dcc.Dropdown(
                        ['ADX-ADXR', 'Aroon', 'CCI', 'DPO', 'EMA', 'HMA', 'MACD_Feats', 'AMA_Diffs', 'KST', 'KST_Diff', 'STC', 'VI_Diff', 'WMA',   # Trend indicators                                                                                    # Trend indicators
                        'AO', 'APO', 'BOP', 'CMO', 'DX', 'STOCH_Diff', 'STOCHF_Diff', 'STOCH_RSI_Diff', 'PVO', 'RSI', 'TSI', 'WILLR',  # Momentum indicators
                        'EOM', 'MFI', 'VRSI', 'FI', 'VPT', 'VWAP',                                                                     # Volume indicators
                        'BB_Diff', 'TRANGE'],                                                                                          # Volatility indicators
                        value = cached_extra_indicators,  # locally stored selections loaded on launching
                        multi=True, id="indicator_dropdown", searchable=False,
                        #style=dropdown_styles_light,
                        persistence=True,                 # keeps selections after only refreshing and not relaunching
                        persistence_type='local',
                        style={'margin-left' : '0.6%', 'margin-right' : '0.6%', 'margin-bottom' : '5px'})

base_feats = ['Log Volume', 'Candlestick Features'] # Base Features besides OHLCV
eng_feats  = ['HLCV MA', 'HLCV Diffs', 'HLCV TLog(Diffs)', 'HLCV Log(Divs)', 
              'HLCV (MA-MA Diffs)', 'HLCV Log(MA-MA Divs)', 'HLCV (now-MA Diffs)', 'HLCV Log(now-MA Divs)',      
              'CSF MA', 'CSF Diffs', 'CSF Log(Divs)', 'CSF (MA-MA Diffs)', 'CSF Log(MA-MA Divs)', 
              'CSF (now-MA Diffs)', 'CSF Log(now-MA Divs)']    
other_feats = ['CumVol Log', 'CumVol (now-later Diffs)', 'CumVol (now-later Divs)', 
               'CumVol TLog(now-later Diffs)', 'CumVol Log(now-later Divs)', 'STD Close', 'Log(STD Volume)']
      
eng_feats  = [[feat + ' (Long)', feat + ' (Short)'] for feat in eng_feats] # Splitting in choice of long- and short-term MAs and Diffs/Divs

all_feats = base_feats + eng_feats + other_feats # list appending
   
feature_dropdown    =  dcc.Dropdown(
                        all_feats,                             
                        value = cached_feature_groups,    
                        multi=True, id="feature_dropdown", searchable=False,
                        #style=dropdown_styles_light,
                        persistence=True,
                        persistence_type='local',
                        style={'margin-left' : '0.6%', 'margin-right' : '6.6%', 'margin-bottom' : '5px'})

worldquant_dropdown =  dcc.Dropdown(['alphas001','alphas002','alphas003','alphas004','alphas005','alphas006','alphas007',
                          'alphas008','alphas009','alphas010','alphas011','alphas012','alphas013','alphas014',
                          'alphas015','alphas016'],     
                          value = cached_worldquant_groups,     
                          multi=True, id="worldquant_dropdown", searchable=False,
                          #style=dropdown_styles_light,
                          persistence=True,
                          persistence_type='local',
                        style={'margin-left' : '0.6%', 'margin-right' : '0.6%', 'margin-bottom' : '5px'})     

filtered_features_dropdown = dcc.Dropdown([],     
                          value = cached_worldquant_groups,     
                          multi=True, id="filtered_features_dropdown", searchable=False,
                          #style=dropdown_styles_light,
                          persistence=True,
                          persistence_type='local',
                        style={'margin-left' : '0.6%', 'margin-right' : '0.6%', 'margin-bottom' : '5px'})    
                          

label_dropdown  = dcc.Dropdown([],          
                          multi=True, id="label_dropdown", searchable=True,
                          disabled = True,
                          #style=dropdown_styles_light
                          )

downsample_freq_dropdown = dbc.DropdownMenu(
            label="Frequency",
            id='downsample_freq_dropdown',
            children=[
                dbc.DropdownMenuItem("1 min"),  # id='1T'),
                dbc.DropdownMenuItem("2 min"),  # id='2T'),
                dbc.DropdownMenuItem("5 min"),  # id='5T'),
                dbc.DropdownMenuItem("10 min"), # id='10T'),
                dbc.DropdownMenuItem("15 min"), # id='15T'),
                dbc.DropdownMenuItem("30 min"), # id='30T'),
                dbc.DropdownMenuItem("1 h"),    # id='1H'),
                dbc.DropdownMenuItem("2 h"),    # id='2H'),
                dbc.DropdownMenuItem("4 h"),    # id='4H'),
                dbc.DropdownMenuItem("12 h"),   # id='12H'),
                dbc.DropdownMenuItem("1 d"),    # id='1D'),
                
            ]), 

label_style = {'margin-bottom' : '10px', 'font-size' : '90%', 'margin-left' : '2px'}

warn_message = f"Your dataset must lie inside 'program/data/datasets/scalping' or 'program/data/datasets/hft' depending on the task."


# ###########################
# ### Dataset Config Card ###
# ###########################

dataset_config_card = dbc.Card([
        dbc.CardHeader(html.H2("Dataset Configuration")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([button_fetch_new_dataset]),            
                dbc.Col([html.Div(
                            "checking ...",
                            style = {'margin-bottom' : '15px', 'font-size' : '90%', 'color' : 'grey'},
                            className="small-text", 
                            id='dataset_file_count_span')
                    ]) 
                ], align='center'),
            
            dbc.Row([
                    dbc.Col([button_select_dataset]),            
                    dbc.Col([
                        html.Div(                   
                            "no file selected",
                            style = {'margin-bottom' : '15px', 'font-size' : '90%', 'color' : 'grey'},
                            className="small-text", 
                            id='filename_span'),
                    ])
                ], align='center'),
            

            
            dbc.Row([
                dbc.Col([html.Div('Coin', 
                        style=label_style,
                        className="small-text")], align='left', style={'margin-left' : '0.6%'}),
                dbc.Col([html.Div('-', 
                        id='label_coin',
                        style=label_style,
                        className="small-text")])], align='left'),
            dbc.Row([
                dbc.Col([html.Div('Stable Coin', 
                        style=label_style,
                        className="small-text")], align='left', style={'margin-left' : '0.6%'}),
                dbc.Col([html.Div('-', 
                        id='label_stable_coin',
                        style=label_style,
                        className="small-text")])], align='left'),
            dbc.Row([
                dbc.Col([html.Div('Frequency', 
                        style=label_style,
                        className="small-text")], align='left', style={'margin-left' : '0.6%'}),
                dbc.Col([html.Div('-', 
                        id='label_fetch_freq',
                        style=label_style,
                        className="small-text")])], align='left'),
            dbc.Row([
                dbc.Col([html.Div('Data Samples', 
                        style=label_style,
                        className="small-text")], align='left', style={'margin-left' : '0.6%'}),
                dbc.Col([html.Div('-', 
                        id='label_sample_count',
                        style=label_style,
                        className="small-text")])], align='left'),
            dbc.Row([
                dbc.Col([html.Div('Input Features', 
                        style=label_style,
                        className="small-text")], align='left', style={'margin-left' : '0.6%'}),
                dbc.Col([html.Div('-', 
                        id='label_feature_count',
                        style=label_style,
                        className="small-text")])], align='left'),

                
            html.Hr(style={'size' : '5px'}),     
            
            dbc.Row([
                dbc.Col([
                    "Technical Indicators⠀", #30 Features
                    dbc.Badge("30", pill=True, color="primary", style={'font-size' : '65%'},
                                className="position-relative top-0 end-60 translate-middle")
                ], style={'margin-left' : '1.2%'}),
            ]),
            
            dbc.Row([
                dbc.Col([indicator_dropdown], style={'margin-left' : '2.5%', 'margin-right' : '2.5%'}),
            ]),
            
            dbc.Row([
                dbc.Col([
                    "AI Features⠀",      #81 Features                # Note: There is no feature engineering for Linear models yet
                    dbc.Badge("82", pill=True, color="primary", style={'font-size' : '65%'},
                            className="position-relative top-0 end-60 translate-middle")
                ], style={'margin-left' : '1.2%'}),
            ]),
            
            dbc.Row([
                dbc.Col([feature_dropdown], style={'margin-left' : '2.5%', 'margin-right' : '2.5%'}),
            ]),
              
            dbc.Row([
                dbc.Col([
                    "WorldQuant Features⠀", # 80 Features
                    dbc.Badge("80", pill=True, color="primary", style={'font-size' : '65%'},
                            className="position-relative top-0 end-60 translate-middle")
                ], style={'margin-left' : '1.2%'}),   
            ]),
            
            dbc.Row([
                dbc.Col([worldquant_dropdown], style={'margin-left' : '2.5%', 'margin-right' : '2.5%'}),   
            ]),
            
           dbc.Row(
                dbc.Col([button_feature_engineering]),
                dbc.Col([button_filter_features]),
                dbc.Col([button_set_lookback_windows])),
           
           
           filtered_features_dropdown,
            
            html.Hr(),
            # dbc.Row([
            #     dcc.Checklist(
            #         ['Scale'],
            #         ['Scale'],
            #         id       = 'scale_checklist',
            #         options  = {'disabled' : True}, # scaling always recommended
            #         inline   = True
            #     )]),
             
            
            dbc.Row([
                dbc.Col([button_use_dataset]),     
                dbc.Col([
                    html.Div([
                        button_download_dataset,
                        dcc.Download(id="dataset_download")
                        ])
                    ]),
            ], align='center'),    
        ])
])

############################
### Dataset Preview Card ###
############################
          
dataset_preview_card = dbc.Card([
        dbc.CardHeader(html.H2("Dataset Preview")),
        dbc.CardBody([
            dbc.Col([
                dbc.Row([data_table]),            
                dbc.Row([feature_graph]),
                dbc.Row([
                    html.Div([

                        dbc.Row([
                            dbc.Col([
                                dbc.Row([html.Div('train :', style=label_style),
                                         dcc.Input(id="input_train", type="number", value=cached_train_perc, step=0.01, persistence=False,
                                                debounce=False, max=1.25, min=0.1, style={'width' : '120px'}),
                                    ])
                                ]),
                            dbc.Col([
                                dbc.Row([html.Div('val   :', style=label_style),
                                         dcc.Input(id="input_val",   type="number", value=cached_val_perc, step=0.01, persistence=False,
                                                debounce=False, max=0.3, min=0.02, style={'width' : '120px'}),
                                    ])
                                ]),
                            dbc.Col([
                                dbc.Row([html.Div('test  :', style=label_style),
                                         dcc.Input(id="input_test",  type="number", value=cached_test_perc, step=0.01, persistence=False,
                                                debounce=False, max=0.5, min=0.02, style={'width' : '120px'}),
                                    ])
                                ]),
                            ], align='center'),
                    ])

                ]),

            ], align='center'),     
        ])
])

layout = dbc.Row([
    dbc.Col([dataset_config_card], width = 4), # 4 ~ 33%
    dbc.Col([dataset_preview_card], width = 6),# 6 ~ 50%
    html.Div('', id='initial_trigger_div'), # always triggers callbacks on page rendering
    html.Div('', id='dummy_output_div'),    # return mandatory outputs to here
    html.Div([dcc.ConfirmDialog(id='warn_dialog',message=warn_message)])
    ], justify="center")


# ########################
# ### Helper Functions ###
# ########################
        
def get_figure(df, params,  
               cols=['Close']):
    """Note: df is passed by reference"""
    t = time.time()
    coin = get_param_by_name(params, 'coin')
    stable_coin = get_param_by_name(params, 'stable_coin')
    fig = go.Figure()

    if len(df) > 1000000:
        # plotting only every 2nd sample for performence (decreasing sampling rate)
        # and keeping 50% of it to show
        decimation = 4
    else:
        decimation = 1
        
        

    
    #df_ind = [i for i in range(len(df.index))][::decimation]
    df_ind = df.index[::decimation]
    for col in cols:
        t2 = time.time()
        fig.add_trace(
            go.Scatter(x=df_ind, y=df[col][::decimation], mode='lines',name=col,
            #line=dict(color=colors[i], width=line_size[i]),
            connectgaps=False, # connect between nan values
        ))
        print('time for adding one trace: ', time.time() - t2)
        
    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.15,
                              xanchor='center', yanchor='top',
                              text=f"Plotted time series of {coin + '/' + stable_coin}",
                              font=dict(family='Arial', size=12, color='rgb(150,150,150)'),
                              showarrow=False))
    
    fig.update_layout(
        xaxis=dict(showline=True, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,255,0.1)',   
                linecolor='rgb(204, 204, 204)', 
                #linewidth=2, 
                showticklabels=True, ticks='outside', 
                tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
        
        yaxis=dict(showline=True, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,255,0.1)',
                linecolor='rgb(204, 204, 204)', 
                #linewidth=2, 
                showticklabels=True, ticks='outside', 
                tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
        #xaxis_title='Time',
        yaxis_title='Features',
        autosize=True,
        showlegend=True,
        plot_bgcolor='white', 
        annotations=annotations
        # margin=dict(autoexpand=False, l=100, r=20, t=110, b=20),
    )

    print('Time for getting figure:', time.time() -t)
    return fig


def get_datatable_data(df : pd.DataFrame, sample_count=100):
    """df is passed by reference as it is a mutable object"""
    # Note: having more than 250 samples inside a DataTable significantly reduces performance!

    #df.reset_index(inplace=True) # include index into table columns
    data           = df[:sample_count].to_dict('records') 
    columns        = [{'id': c, 'name': c, 'type': 'numeric', 
                       'format':Format(precision=4, scheme=Scheme.fixed),
                       'selectable' : True } for c in df.columns]
    #tooltip_header = {i: i for i in df.columns}
    return data, columns
     
def get_model_config_href(params : dict):
    """ gets the page's path inside ExperATAI\program\pages
        for the 'Use Dataset' button. 
    """
    task     = get_param_by_name(params, 'experiment_task_name')
    model_config_path = get_path_by_path_name('model_config_dir_path')
    pages_base_path   = get_path_by_path_name('pages_dir_path')
    
    rel_path = model_config_path.replace(pages_base_path, '') # --> '/model_config'
    if task == 'LTSF':
        return rel_path + '/model_config_ltsf'
    elif task == 'RLT':
        return rel_path + '/model_config_rl'
    
def get_dataset_loc(params):
    is_hft_dataset = get_param_by_name(params, 'is_hft_dataset')
    if is_hft_dataset:
        dataset_loc = '/hft' 
    else:
        dataset_loc = '/scalping' 
    dataset_path = get_path_by_path_name('dataset_dir_path') + dataset_loc
    return dataset_path
    
def check_dataset_validity(params : dict, filename : str):
    """Check if a selected dataset is from the rigth directory"""
    # datasets are either in data/hft or data/scalping
        
    dataset_path = get_dataset_loc(params)
    is_valid = filename in os.listdir(dataset_path)     
    
    return is_valid

def get_data_save_path(params, filename):
    """ # --> G:\ExperATAI\program\data\datasets\scalping """
    dataset_path = get_dataset_loc(params)
    dataset_path = dataset_path + '/' + filename
    return dataset_path 

def get_dataset_file_count(params):
    dataset_path = get_dataset_loc(params) 
    files = [i for i in os.listdir(dataset_path)]
    dataset_file_count = len(files)
    return dataset_file_count
    

def get_dataset_class(params):
    task = get_param_by_name(params, 'experiment_task_name')
    if task == 'LTSF':
        return Crypto_LTSF_Dataset
    elif task == 'RLT':
        return Crypto_RL_Dataset
    
def get_dataset_df(params):
    t = time.time()
    cls_name = get_dataset_class(params)  
    experiment_cls = Experiment_Config()
    experiment_cls.set_params(params)
    experiment_cls.build_dataset(cls_name)
    print('Time for dataset creation:', time.time() - t)
    return experiment_cls.experiment_data['dataset'].df

def get_is_dropdown_disabled(params):
    d_ind, d_feat, d_world = False, False, False
    is_hft_dataset = get_param_by_name(params, 'is_hft_dataset')
    task           = get_param_by_name(params, 'experiment_task_name') 
    
    if is_hft_dataset:   # no indicators and worldquant features for hft! (no recorded volume and open)
        d_ind = True
        d_world = True
    if task == 'LTSF':   # LTSF should only forecast oHLCV series with feature engineering 
                         #   (most indicators and worldquant features are very uninformative for that)

        d_ind = True
        d_world = True
        pass
    if task == 'RLT':    # RLT should fell trading decisions based on indicators, worldquant features or feature engineering 
        pass
    
    return d_ind, d_feat, d_world
    
########################
### Button Callbacks ###
########################   

#### Callbacks for dataset selection ####

@callback( 
    [Output('warn_dialog',         'displayed')],
    [Input('button_select_dataset', 'filename')],
     State('param_storage'           , 'data'),
     prevent_initial_call=True)
def warn_on_invalid_dataset(filename, params): 
    if filename != None:
        selection_is_valid = check_dataset_validity(params, filename)
        return [not selection_is_valid]
    else:
            raise PreventUpdate

@callback( 
    [Output('filename_span',       'children'),
     Output('label_coin',          'children'),
     Output('label_stable_coin',   'children'),
     Output('label_fetch_freq',    'children')],
    [Input('button_select_dataset', 'filename')],
     State('param_storage'           , 'data'),
     prevent_initial_call=True)
def update_labels_on_selection(filename, params):  
    selection_is_valid = check_dataset_validity(params, filename)
    if selection_is_valid:
        vals = filename.split('_') # ALGO_USDT_1m_2020-01-2022-06.csv --> ['ALGO', 'USDT', '1m', '2020-01-2022-06.csv']
        save_path = get_data_save_path(params, filename)
        return filename, vals[0], vals[1], vals[2]
    else:
        raise PreventUpdate
    
@callback( 
    [Output('button_use_dataset',         'disabled'),
     Output('button_use_dataset',         'href'),
     Output('button_feature_engineering', 'disabled', allow_duplicate=True),
     Output('button_download_dataset'   , 'disabled')],
    [Input('button_select_dataset', 'filename')],
     State('param_storage'           , 'data'),
     prevent_initial_call=True)
def update_buttons(filename, params):  
    selection_is_valid = check_dataset_validity(params, filename)
    if selection_is_valid:
        return False, get_model_config_href(params), False, False
    else:
        raise PreventUpdate

@callback( 
    [Output('param_storage'      ,  'data',   allow_duplicate=True),
     Output('feature_graph'      ,  'figure', allow_duplicate=True),
     Output('data_table'         , 'data',    allow_duplicate=True),
     Output('data_table'         , 'columns', allow_duplicate=True),
     Output('label_feature_count', 'children', allow_duplicate=True),
     Output('label_sample_count', 'children', allow_duplicate=True)],
    [Input('button_select_dataset', 'contents'),
     Input('button_select_dataset', 'filename')],
    [State('param_storage'           , 'data')],
     prevent_initial_call=True)
def update_params_and_data(contents, filename, params):  
    """ Callback that updates params on selecting dataset 
        and then assigns the new data to the datatable and the graph."""
    selection_is_valid = check_dataset_validity(params, filename)
    if selection_is_valid:     
        # Set params   
        vals = filename.split('_') # ALGO_USDT_1m_2020-01-2022-06.csv --> ['ALGO', 'USDT', '1m', '2020-01-2022-06.csv'] 
        new_params = {'coin'            : vals[0], 
                      'stable_coin'     : vals[1], 
                      'recorded_freq'   : vals[2],
                      'downsample_freq' : None,
                      'data_file_path'  : get_data_save_path(params, filename)}
        params = update_dict_wrapper(params, new_params)
               
        # Initialize dataset from params
        df = get_dataset_df(params)
        # Get plot and datatable dataset
        fig = get_figure(df, params)
        data, columns = get_datatable_data(df)
        feature_count = len(df.columns)
        sample_count  = len(df)
        return params, fig, data, columns, feature_count, sample_count
    else:
        raise PreventUpdate
   
   
### Callbacks for feature engineering ###

"""  Note: Feature engineering is also done when selecting or changing the dataset in any way, 
           if any feature groups were selected. 
           Feature engineering can be done explicitely when pressing 'Calculate Features' button 
           after selecting feature groups.
"""  
@callback(
    [
     Output('button_feature_engineering' , 'disabled', allow_duplicate=True),
     Output('label_feature_count'        , 'children', allow_duplicate=True),
     Output('feature_graph'              , 'figure'  , allow_duplicate=True),
     Output('data_table'                 , 'data'    , allow_duplicate=True),
     Output('data_table'                 , 'columns' , allow_duplicate=True)],
    [Input('button_feature_engineering'  , 'n_clicks')],
     State('param_storage'               , 'data'),
    prevent_initial_call=True)
def feature_engineering(n1, params):
    # Initialize dataset from params
    df = get_dataset_df(params)
    # Get plot and datatable dataset
    col_count = len(df.columns)
    fig = get_figure(df, params)
    data, columns = get_datatable_data(df)
    return True, col_count, fig, data, columns 

   
### Callbacks for 'Use Dataset' ###     
        
@callback(
     Output('dummy_output_div'  , 'children'),
    [Input('button_use_dataset' , 'n_clicks')],
     State('param_storage'      , 'data'),
     prevent_initial_call=True)
def chache_params_on_use_dataset(n1, params): 
    cache_params(params, verbose=True)
    return [None] 
    

### Callbacks for 'Use Dataset' ###     
    
@callback( 
    Output("dataset_download", "data"),
    Input("button_download_dataset", "n_clicks"),
    State('param_storage'   ,'data'),
    prevent_initial_call=True)
def download_dataset(n_clicks, params): # always download pkl to avoid up to 10 min saving time!
    file_name = params['coin'] + '_' + params['stable_coin'] + params['coin'] + '.pkl'
    return dcc.send_data_frame(local_params.experiment_data['dataset'].to_pickle, file_name)



#######################
### Input Callbacks ###
#######################

@callback(
    [Output('param_storage'  ,'data', allow_duplicate=True),
     Output('input_train'    ,'style'),
     Output('input_val'      ,'style'),
     Output('input_test'     ,'style')],
    [Input('input_train'     ,'value'),
     Input('input_val'       ,'value'),
     Input('input_test'      ,'value')],
    [State('param_storage'   ,'data'),
     State('input_train'     ,'style')],
     prevent_initial_call=True)
def set_train_test_split(train, val, test, params, input_styles):
    # Only update params on valid train-val-test splits
    if None in [train, val, test]:
        raise PreventUpdate 
    input_styles['border-width'] = '1px'
    if np.sum([train, val, test]) == 1.0: 
        new_params = {'train_perc'     : train,
                      'val_perc'       : val,
                      'test_perc'      : test}
        input_styles['border-color'] = ''
        params = update_dict_wrapper(params, new_params)
        return params, input_styles, input_styles, input_styles
    else:
        input_styles['border-color'] = 'orange'
        return params, input_styles, input_styles, input_styles

    
@callback(
    [Output('param_storage'              ,'data',     allow_duplicate=True),
     Output('button_feature_engineering' ,'disabled', allow_duplicate=True)],
    [Input('indicator_dropdown' ,'value'),
     Input('feature_dropdown'   ,'value'),
     Input('worldquant_dropdown','value')],
    [State('param_storage'      ,'data'),
     State('filename_span'      ,'children')],
     prevent_initial_call=True)
def update_feature_selection(ind, feats, world, params, fn_text): # Inputs: String lists
    ind   = [] if ind   == None else ind
    feats = [] if feats == None else feats
    world = [] if world == None else world
    new_params = {'extra_indicators'  : ind,
                  'feature_groups'    : feats,
                  'worldquant_groups' : world}
    params = update_dict_wrapper(params, new_params)
    disabled = sum([len(ind), len(feats), len(world)]) == 0 or fn_text == 'no file selected'
               
    return params, disabled

@callback( 
    Output('feature_graph', 'figure'), 
    Input('data_table', 'selected_columns'),
    State('param_storage'      ,'data'),
    prevent_initial_call=True)
def update_data_plot_on_col_selection(sel_cols, params):
    #TODO make callback on button!
    print('sel_cols', sel_cols)
    df = get_dataset_df(params)
    fig = get_figure(df, params, cols=sel_cols)  
    return fig

##############################
### Theme Switch Callbacks ###
##############################

dropdown_styles_dark = {'margin-top'    : '5px',
                        'margin-bottom' : '10px', 
                        'color'         : '#333333', #text color,
                        'border-color': '#444444',
                        'background-color': '#353535'
                }

dropdown_styles_light = {'margin-top'    : '5px',
                        'margin-bottom' : '10px', 
                        'color'         : 'grey', #text color,
                }

@callback(
    [Output("indicator_dropdown", "style"),
     Output("feature_dropdown", "style"),
     Output("worldquant_dropdown", "style")],
    [Input(ThemeSwitchAIO.ids.switch("theme_switch"), "value")]
    )
def switch_dropdown_themes(toggle):
    is_dark = toggle
    if is_dark:
        return [dropdown_styles_dark]*3
    else:
        return [dropdown_styles_light]*3
  
@callback(
    [Output("data_table", "style_header"),
     Output("data_table", "style_data"),],
    [Input(ThemeSwitchAIO.ids.switch("theme_switch"), "value")]
    )
def switch_dropdown_themes(toggle):
    is_dark = toggle
    if is_dark:
        style_header = {'backgroundColor': '#222222',
                        'color'          : 'white',
                        'fontWeight'     : 'bold',
                        'font-size'      : '80%',
                        'text-align'     : 'center',
                        'border'         : '1px solid #333333',
                        }
        style_data   = {'color'          : 'white',
                        'backgroundColor': '#353535',
                        'font-size'      : '80%',
                        'border'         : '1px solid #404040',
                        'border-color'   : 'white'}
    else:
        style_header = {'fontWeight'     : 'bold',
                        'font-size'      : '80%',
                        'text-align'     : 'center'}
        style_data   = {'font-size'      : '80%',} #default values
        
    return style_header, style_data
    
     
  # style_header={
                    #     'backgroundColor': 'white',
                    #     'color': 'black',
                    #     'fontWeight': 'bold',
                    #     'border': '1px solid blue'
                    # },
                    
                    # style_data={
                    #     'color': 'black',
                    #     'backgroundColor': 'white',
                    #     'border': '1px solid blue'
                    # },
  
  
# @callback(
#     [Output('feature_graph'  ,'figure')],
#     [Input('input_train'     ,'value')],
#      State('param_storage'   ,'data'))
# def switch_graph_themes(toggle):
#     d_ind, d_feat, d_world = False, False, False
#     is_hft_dataset = get_param_by_name(params, 'is_hft_dataset')
#     task = get_param_by_name(params, 'experiment_task_name')
    
#     if is_hft_dataset:   # no indicators and worldquant features for hft!
#         d_ind = True
#         d_world = True
#     if task == 'LTSF':   # no contraints yet
#         pass
#     if task == 'RLT':    # no contraints yet
#         pass
    
#     p_ind   = '-' if d_ind   else 'Select...'
#     p_feat  = '-' if d_feat  else 'Select...' 
#     p_world = '-' if d_world else 'Select...'
     
#     return d_ind, d_feat, d_world, p_ind, p_feat, p_world  
    
    
#TODO: check if displayModeBar and displaylogo should be disabled

#########################
### Initial Callbacks ###
#########################
""" 
    Called once the page is loaded  
    Fills Web components with default or cached params 
"""
    
@callback(
    [Output('indicator_dropdown'  ,'disabled'),
     Output('feature_dropdown'    ,'disabled'),
     Output('worldquant_dropdown' ,'disabled'),
     Output('indicator_dropdown'  ,'placeholder'),
     Output('feature_dropdown'    ,'placeholder'),
     Output('worldquant_dropdown' ,'placeholder'),
     Output('indicator_dropdown'  ,'persistence'),
     Output('feature_dropdown'    ,'persistence'),
     Output('worldquant_dropdown' ,'persistence')],
    [Input('initial_trigger_div'  ,'children')],
     State('param_storage'   ,'data'))
def disable_dropdowns(_temp, params):
    """ Disables dropdowns for certain tasks """
    d_ind, d_feat, d_world = get_is_dropdown_disabled(params)
    
    p_ind   = '-' if d_ind   else 'Select...'
    p_feat  = '-' if d_feat  else 'Select...' 
    p_world = '-' if d_world else 'Select...'
    
    # avoid keeping selected values for disabled dropdowns on page refresh if they are disabled
    # (selections are kept when persistence of dropdown = True)
    pers_ind, pers_feat, pers_world = np.invert([d_ind, d_feat, d_world]) 
   
    return d_ind, d_feat, d_world, p_ind, p_feat, p_world, pers_ind, pers_feat, pers_world

@callback(
    [Output('dataset_file_count_span'  , 'children'),
     Input('initial_trigger_div'  ,'children')],
     State('param_storage'   ,'data')
)
def set_dataset_file_count_span(_temp, params):
    file_count   = get_dataset_file_count(params)
    task = get_param_by_name(params, 'experiment_task_name')
    return [f'{file_count} files for {task}']


# TODO theme style for graph    

#TODO cache on use_dataset
#TODO feature engineering

# Note: https://plotly.com/python/line-charts/
    # add traces on relative growth
    
# Note: https://plotly.com/python/line-charts/
    # add background lines for MAE on LTSF plots
    
# Note: Gant charts: https://plotly.com/python/gantt/

# Note: scatter plots for correlation

# Note: parallel coordinates: https://plotly.com/python/parallel-coordinates-plot/
# Note: parallel categories:  https://plotly.com/python/parallel-categories-diagram/

# Snippet: get_col_names: for i in range(len(fig.data)):
    #                         print(fig.data[i]['legendgroup'])
    
    #[fig.data[i]['legendgroup'] for i in range(len(fig.data))]
    
#TODO on col select: change graph by its fig data and sel columns
    
#TODO fill labels on every dataset change
    # clear on change if label not in columns

#TODO label dropdown


# TODO RAM, CPU, GPU Gauge (color green to red), ausfahrmenü

#TODO 5 decimals in table

#TODO table fields not selectable

#TODO initial label filling

#TODO check download

#TODO dataset loading feedback

#TODO add 'Close' to selected columns after selecting dataset

#TODO add train_test_val samples as choice (take 200.000 newest train samples) --> then train = 0.3
# add triple slider parallel to graph

# Note: Improving plotly performance: https://www.somesolvedproblems.com/2018/07/how-do-i-make-plotly-faster.html

# Note 3rd place solution lGBM: https://www.kaggle.com/code/sugghi/training-3rd-place-solution
#TODO remove featureengineering from LTSF; no --> turn it into label calculation



#TODO reimplement worldquant, Indicators for ltfs
    # re-implement all for RL scalping
    
#TODO re-implement downsampling, clipping --> Misc Card