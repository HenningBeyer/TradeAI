""" File for reading in datasets and doing preprocessing with clipping, downsampling, train-test-splitting and feature engineering.
        These classes are used primatily for visualization and saving before training and serve common preprocessing methods in 
        simpler dataframe form.
    
    Once doing training of models, only the train, test or validation splits of the dataframe are used as torch datasets.
"""

from experatai_backend.worldquants_101alphas import Alphas_101
from experatai_backend.hft_data_fetching import Live_HFT_Sample_Fetcher
from experatai_backend.utils import tlog

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import ta
import talib
import time

warnings.filterwarnings('ignore')

class Base_Crypto_Dataset(): 
    """ Dataset class responsible for loading and managing the dataset contained in a file.
            Does offer functionality for loading, saving and clipping a crypto dataset.
        
        More specific downsampling and feature engineering is done in Crypto_Scalping_Dataset and Crypto_Scalping_Dataset
        
        Base_Crypto_Dataset is typically used for pre-training and a Data_Stream class 
            for downstream application and fine-tuning.
    """

    def __init__(self,
                 params):  
                 
        self.coin             = params['coin'           ]         
        self.stable_coin      = params['stable_coin'    ]
        self.downsample_freq  = params['downsample_freq'] # in min
        self.recorded_freq    = params['recorded_freq'  ]
        self.data_filepath    = params['data_filepath' ]
        self.df               = self._load_dataset()         
                                        
        self.train_perc    = params['train_perc' ]
        self.val_perc      = params['val_perc'   ]
        self.test_perc     = params['test_perc'  ]
        self.scale         = params['scale'      ]       
        self.start_date    = params['start_date' ]  # assumed format YYYY-MM-DD; YYYY-MM-DD HH:mm:ss will work too 
        self.end_date      = params['end_date'   ]  # dates outside of the df time index interval do also work
        self.target_cols   = params['target_cols']
        self.scaler        = StandardScaler()
        self.scaler_fitted = False
         
        assert np.sum([self.train_perc, self.val_perc, self.test_perc]) == 1.0
        if self.start_date != None or self.end_date != None:
           self._clip_dataset()
           
        self.used_freq        = self.recorded_freq is self.downsample_freq == None else self.downsample_freq
        self.dataset_filename = os.path.basename(params['data_filepath'])
        
    def _clip_dataset(self): # does also work when one or both dates are equal to None
        self.df = self.df[self.start_date : self.end_date]
            
    def _load_dataset(self):
            if '.pkl' in self.data_filepath: #preferred
                        df = pd.read_pickle(self.data_filepath, index_col='Time')
                        
            elif '.csv' in self.data_filepath: # slower
                        df = pd.read_csv(   self.data_filepath, index_col='Time')
        
            df.index = pd.to_datetime(df.index)       # Make time index instead of string index
            df.fillna(method='ffill', inplace=True)   # Fill potential NaN values
            return df    

    def save_df_to_csv(self, fn : str): #If inherited it will save also engineered data
        abs_dir_path = os.path.dirname(os.path.abspath(__file__))
        self.df.to_csv(abs_dir_path + '/data/saved_datasets/' + fn + '.csv')  
                                        # Extremely slow for very large files > 1 GB
                                        # Data size: 4.2 GB 
                                        # Saving time ~ 10 min
                                        # Reading time ~ 3 min
        
    def save_df_to_pickle(self, fn : str): #If inherited it will save also engineered data
        abs_dir_path = os.path.dirname(os.path.abspath(__file__))
        self.df.to_pickle(abs_dir_path + '/data/saved_datasets/' + fn + '.pkl')   
                                        # A lot faster writing and saving; 
                                        # Data size: 2.6 GB compared to 4.2 GB 
                                        # Saving time ~ 5s
                                        # Reading time ~ 15s

                
               
    def get_train_test_split_data(self, flag):           
        """ Used for creating torch datasets and dataloaders with either train, test or val data. """    
            
        num_train           = int(len(self.df) * self.train_perc)        
        num_val             = int(len(self.df) * self.val_perc)
        num_test            = int(len(self.df) * self.test_perc)
        
        assert np.array([c in self.df.columns for c in self.target_cols]).all(), 'All target features should always be in the input data'

        if flag in ['train' ]:
            return self.df.reset_index(drop=True)[0         : num_train]
        elif flag in ['val' ]:
            return self.df.reset_index(drop=True)[num_train : num_test ]
        elif flag in ['test']:
            return self.df.reset_index(drop=True)[num_test  : len(df)  ]  
    
    def get_train_scaler(self):
        num_train = int(len(self.df) * self.train_perc)  
        data = self.df.reset_index(drop=True)[0         : num_train]
        self.scaler.fit(self.df.reset_index(drop=True)[0 : num_train].to_numpy())
        return self.scaler
        
class Crypto_OHLCV_Dataset(Base_Crypto_Dataset):
    """ Class providing its own downsampling and feature engineering. """
    
    def __init__(self, params):                                       
        super(Crypto_TSF_Dataset, self).__init__(params)
        if self.downsample_freq != None:
            self._downsample_ohlcv_df()
            
        self.feature_engineering = Orderbook_Feature_Engineering(feature_groups    = params['feature_groups'],
                                                                 extra_indicators  = params['extra_indicators'], 
                                                                 worldquant_groups = params['worldquant_groups'],
                                                                 filtered_features = params['filtered_features'])
        self.feature_engineering.full_engineering(self.df)  
 

    def _downsample_ohlcv_df(self): 
        assert len(self.df.columns) == 5 # only OHLCV data
        
        # valid frequencies should range from 1T up to 1D for scalping  
        limit_freq = '1D'
        recorded_test_range   = pd.date_range('11-05-2023', '11-05-2023 00:01:00', freq=self.recorded_freq)
        downsample_test_range = pd.date_range('11-05-2023', '11-05-2023 00:01:00', freq=self.downsample_freq)
        min_freq_test_range   = pd.date_range('11-05-2023', '11-05-2023 00:01:00', freq=limit_freq)
        assert len(min_freq_test_range) < len(downsample_test_range) < len(recorded_test_range), f"downsample_freq must be between {self.recorded_freq} and {limit_freq}."
        
        ohlcv_dict = {
            'Open':'first',
            'High':'max',
            'Low':'min',
            'Close':'last',
            'Volume':'sum'
        }
        self.df = self.df.resample(self.downsample_freq).apply(ohlcv_dict)


class Crypto_Orderbook_Dataset(Base_Crypto_Dataset):
    """ Class providing its own downsampling and feature engineering. """
    
    def __init__(self, params):       
        
        super(Crypto_RL_Dataset, self).__init__(params)
        if self.downsample_freq != None:
            self._downsample_orderbook_df()
            
        self.feature_engineering = OHLCV_Feature_Engineering(feature_groups    = params['feature_groups'],
                                                             filtered_features = params['filtered_features'])
        self.feature_engineering.full_engineering(self.df) 
                
        def _downsample_orderbook_df(self):   
        
        limit_freq = '1T'
        recorded_test_range   = pd.date_range('11-05-2023', '11-05-2023 00:01:00', freq=self.recorded_freq)
        downsample_test_range = pd.date_range('11-05-2023', '11-05-2023 00:01:00', freq=self.downsample_freq)
        min_freq_test_range   = pd.date_range('11-05-2023', '11-05-2023 00:01:00', freq=limit_freq)
        assert len(min_freq_test_range) < len(downsample_test_range) < len(recorded_test_range), f"downsample_freq must be between {self.recorded_freq} and {limit_freq}."

        self.df = self.df.resample(self.downsample_freq).apply('last')  
        
class OHLCV_Feature_Engineering():
    """ A class that does feature engineering for OHLCV datasets may used by dataset and datastream classes. """     
    def __init__(self, feature_groups=[], extra_indicators=[], worldquant_groups=[], filtered_features=[]):
        # This is a class that only hold methods and gets called by other classes
        
        self.feature_groups      = feature_groups
        self.extra_indicators    = extra_indicators
        self.worldquant_groups   = worldquant_groups
        self.filtered_features   = filtered_features

    def full_engineering(self, df):
        
        df = self._feature_engineering_scalping(df)
        df = self._indicator_engineering_scalping(df)
        df = self._alpha101_engineering_scalping(df)
        
        filtered_features = set(filtered_features).intersection(df.columns) # can only drop columns existing in df
        df.drop(filtered_features, inplace=True, axis=1)
        
        df.fillna(method='ffill', inplace=True) #filling occasional NaN-values for log features in their columns
        df.dropna(inplace=True)
        
        return df
        
        
    def _feature_engineering_scalping(self, df):   
        """ All features of this functions are expected to be mostly suitable for
                time series forecasting but also for RL trading.
                For time series forecasting, generally more raw and abstract features are prefered than for RLT.
             
            Volume features are mostly normazlied by Log(a).
                Differences between Volumes are normalized by TLog(a-b), whereas most other differences can remain unnormalized. 
                Divisions (a/b), Differences (a-b) are important to RL systems as they provide real absolute and relative relations.
                Log(a/b), TLog(a-b) are easier to learn for momentum estimation due to reduced noise.
                All variants are useful but amount to a lot of features: Watch for overfitting; chose fewer!
            
            Features for shifting values are not part of feature engineering, since most models generate them with looking back at past values.
                MA of differences or divisions are generally very misleading and thus not part inside this script.
            
            There are a more than 600 features (~300 HLCV Features, and ~300 Candlestick Features)
                By chosing either short or long windows, these amounts will be approximately halved.
                All MA/momentum features get split into short and long to allow this flexible engineering. 
        """
        
        MA_windows       = [ 6, 15, 30, 60, 120, 300] # Windows intended for 1 min frequency and not for 1 s frequency!
        MA_windows_short = [ 6, 15, 30] 
        MA_windows_long  = [60,120,300]
        
        min_max_windows       = MA_windows
        min_max_windows_short = MA_windows_short
        min_max_windows_long  = MA_windows_long
        
        momentum_shift_vals =       [ 1, 2, 3, 4, 5, 6, 15, 30, 60, 120, 300]
        momentum_shift_vals_short = [ 1, 2, 3, 4, 5, 6] 
        momentum_shift_vals_long  = [15,30,60,120,300 ] # long term momentum better for RL
        
        cum_volume_windows = [6,15,30,60,120,300]
        std_windows        = [  15,30,60,120    ]
        
        ### Split features into short-term and long-term.
        def _get_win_by_name(self, mode, feat_name):
            name_long  = feat_name + ' (Long)'
            name_short = feat_name + ' (Short)'
            if mode == 'MA':
                short_wins = MA_windows_short          if name_short in self.feature_groups else []
                long_wins  = MA_windows_long           if name_long  in self.feature_groups else []
            elif mode == 'momentum':
                short_wins = momentum_shift_vals_short if name_short in self.feature_groups else []
                long_wins  = momentum_shift_vals_long  if name_long  in self.feature_groups else []
            return short_wins + long_wins # list appending
        
        self.feature_groups = np.unique([feat.replace(' (Long)', '').replace(' (Short)', '') for feat in self.feature_groups]) # base names of self.feature_groups
        df['LogVolume'] = np.log(df['Volume']+1)
        
        """"""""""""""""""""""          
        ### HCLV Features ###
        """"""""""""""""""""""
        
        feats = ['High', 'Low', 'Close', 'LogVolume']
        
        ### MA Features
        if 'HLCV MA' in self.feature_groups:
            MA_wins = self._get_win_by_name(mode='MA', 'HLCV MA')
            for feat in feats:
                for w in MA_wins:
                    df[f'(MA_{w})_{feat}'] = df[feat].rolling(window=w).mean()

        ### Momentum Features
        if 'HLCV Diffs'  self.feature_groups: # Absolute momentum
            mom_wins = self._get_win_by_name(mode='momentum', 'HLCV Diffs')
            for feat in feats:
                for msv in mom_wins:
                    df[f'(Diff_{msv})_{feat}'] = df[feat].diff(msv)
                    
        if 'HLCV TLog(Diffs)'  self.feature_groups: # Absolute momentum
            mom_wins = self._get_win_by_name(mode='momentum', 'HLCV Diffs')
            for feat in feats:
                for msv in mom_wins:
                    df[f'TLog(Diff_{msv})_{feat}'] = tlog(df[feat].diff(msv))
            
        if 'HLCV Log(Divs)' in self.feature_groups: # Log Returns
            mom_wins = self._get_win_by_name(mode='momentum', 'HLCV Log(Divs)')
            for feat in feats:
                for msv in mom_wins:
                    df[f'Log(Div_{msv})_{feat}'] = np.log(df[feat]/df[feat].shift(msv))
                    
                    
        ### HCLV MA Momentum Features          
        if 'HLCV (MA-MA Diffs)' in self.feature_groups:
            MA_wins = self._get_win_by_name(mode='MA', 'HLCV (MA-MA Diffs)') # Difference of a MA to next, second next and third next MAs 
            for feat in feats:
                for i in range(len(MA_wins)-1):
                    df[f'(MA_{wins[i]}-MA_{wins[i+1]})_{feat}' ] = df[feat].rolling(window=wins[i]).mean() - df[feat].rolling(window=wins[i+1]).mean()
                for i in range(len(MA_wins)-2):
                    df[f'(MA_{wins[i]}-MA_{wins[i+2]})_{feat}' ] = df[feat].rolling(window=wins[i]).mean() - df[feat].rolling(window=wins[i+2]).mean()
                for i in range(len(MA_wins)-3):
                    df[f'(MA_{wins[i]}-MA_{wins[i+3]})_{feat}' ] = df[feat].rolling(window=wins[i]).mean() - df[feat].rolling(window=wins[i+3]).mean()
                    
        if 'HLCV (now-MA Diffs)' in self.feature_groups:
            MA_wins = self._get_win_by_name(mode='MA', 'HLCV (now-MA Diffs)')
            for feat in feats:
                for w1 in MA_wins[:]:
                    df[f'(now-MA_{w1})_{feat}'] = df[feat] - df[feat].rolling(window=w1).mean() 
 
        if 'HLCV Log(MA-MA Divs)' in self.feature_groups:
            MA_wins = self._get_win_by_name(mode='MA', 'HLCV Log(MA-MA Divs)')
            for feat in feats:
                for i in range(len(MA_wins)-1):
                    df[f'Log(MA_{wins[i]}/MA_{wins[i+1]})_{feat}'] = np.log(df[feat].rolling(window=wins[i]).mean() / df[feat].rolling(window=wins[i+1]).mean())
                for i in range(len(MA_wins)-2):
                    df[f'Log(MA_{wins[i]}/MA_{wins[i+2]})_{feat}'] = np.log(df[feat].rolling(window=wins[i]).mean() / df[feat].rolling(window=wins[i+2]).mean())
                for i in range(len(MA_wins)-3):
                    df[f'Log(MA_{wins[i]}/MA_{wins[i+3]})_{feat}'] = np.log(df[feat].rolling(window=wins[i]).mean() / df[feat].rolling(window=wins[i+3]).mean())
                                                            
        if 'HLCV Log(now-MA Divs)' in self.feature_groups:
            MA_wins = self._get_win_by_name(mode='MA', 'HLCV Log(now-MA Divs)')
            for feat in feats:
                for w1 in MA_wins:
                    df[f'Log({feat}/MA_{w1})'] =  np.log(df[feat] / df[feat].rolling(window=w1).mean())

        ### Min-Max-Momentum Features
        
        ## Min
        if 'HLCV (now-Min Diffs)' in self.feature_groups:
            min_max_wins = self._get_win_by_name(mode='MA', 'HLCV (now-Min Diffs)')
            for feat in feats:
                for w1 in min_max_wins:
                    df[f'(now-Min_{w1})_{feat}'] = df[feat] - df[feat].rolling(window=w1).min()
                                                             
        if 'HLCV TLog(now-Min Diffs)' in self.feature_groups:
            min_max_wins = self._get_win_by_name(mode='MA', 'HLCV TLog(now-Min Diffs)')
            for feat in feats:
                for w1 in min_max_wins:
                    df[f'TLog(now-Min_{w1})_{feat}'] = tlog(df[feat] - df[feat].rolling(window=w1).min())
                        
        if 'HLCV Log(now-Min Divs)' in self.feature_groups:
            min_max_wins = self._get_win_by_name(mode='MA', 'HLCV Log(now-Min Divs)')
            for feat in feats:
                for w1 in min_max_wins:
                    df[f'Log(now/Min_{w1})_{feat}'] = np.log(df[feat] / df[feat].rolling(window=w1).min()) 
        ## Max
        if 'HLCV (now-Max Diffs)' in self.feature_groups:
            min_max_wins = self._get_win_by_name(mode='MA', 'HLCV (now-Max Diffs)')
            for feat in feats:
                for w1 in min_max_wins:
                    df[f'(now-Max_{w1})_{feat}'] = df[feat] - df[feat].rolling(window=w1).max()
                                                             
        if 'HLCV TLog(now-Max Diffs)' in self.feature_groups:
            min_max_wins = self._get_win_by_name(mode='MA', 'HLCV TLog(now-Max Diffs)')
            for feat in feats:
                for w1 in min_max_wins:
                    df[f'TLog(now-Max_{w1})_{feat}'] = tlog(df[feat] - df[feat].rolling(window=w1).max())
                        
        if 'HLCV Log(now-Max Divs)' in self.feature_groups:
            min_max_wins = self._get_win_by_name(mode='MA', 'HLCV Log(now-Max Divs)')
            for feat in feats:
                for w1 in min_max_wins:
                    df[f'Log(now/Max_{w1})_{feat}'] = np.log(df[feat] / df[feat].rolling(window=w1).max()) 

                              
        """"""""""""""""""""""""""""   
        ### Candlestick Features ###
        """"""""""""""""""""""""""""
        # No TLog for CSF as differences are always small
         
        ### Basic Candlestick Features    # deleted if not 'Candlestick Features' in feature_groups --> simplifies code a lot
        df['Close-High'] = df['Close'] - df['High']
        df['Close-Low' ] = df['Close'] - df['Low' ]
        df['High-Low'  ] = df['High' ] - df['Low' ]          
        df['UShadow'   ] = df['High' ] - np.maximum(df['Close'], df['Open'])
        df['BShadow'   ] = np.minimum(df['Close'], df['Open']) - df['Low']
           
        feats = ['Close-High', 'Close-Low', 'High-Low', 'UShadow', 'BShadow']
           
        ### MA Features
        if 'CSF MA' in self.feature_groups:
            MA_wins = self._get_win_by_name(mode='MA', 'CSF MA')
            for feat in feats:
                for w1 in MA_wins[:]:
                    df[f'(MA_{w1})_{feat}'] = (df[feat]).rolling(window=w1).mean()
          
        ### Momentum Features              
        if 'CSF Diffs'  in self.feature_groups: # Absolute momentum
            mom_wins = self._get_win_by_name(mode='momentum', 'CSF Diffs')
            for feat in feats:
                for msv in mom_wins:
                    df[f'(Diff_{msv})_{feat}'] = df[feat].diff(msv)
                    
        if 'CSF Log(Divs)' in self.feature_groups: # Log Returns
            mom_wins = self._get_win_by_name(mode='momentum', 'CSF Log(Divs)')
            for feat in feats:
                for msv in mom_wins:
                    df[f'Log(Div_{msv})_{feat}'] = np.log(self.df[feat]/self.df[feat].shift(msv))
                
        ### MA Momentum Features    
        if 'CSF (MA-MA Diffs)' in self.feature_groups:
            MA_wins = self._get_win_by_name(mode='MA', 'CSF (MA-MA Diffs)') 
            for feat in feats:
                for i in range(len(MA_wins)-1):
                    df[f'(MA_{wins[i]}-MA_{wins[i+1]})_{feat}' ] = df[feat].rolling(window=wins[i]).mean() - df[feat].rolling(window=wins[i+1]).mean()
                for i in range(len(MA_wins)-2):
                    df[f'(MA_{wins[i]}-MA_{wins[i+2]})_{feat}' ] = df[feat].rolling(window=wins[i]).mean() - df[feat].rolling(window=wins[i+2]).mean()
                for i in range(len(MA_wins)-3):
                    df[f'(MA_{wins[i]}-MA_{wins[i+3]})_{feat}' ] = df[feat].rolling(window=wins[i]).mean() - df[feat].rolling(window=wins[i+3]).mean()
                    
        if 'CSF Log(MA-MA Divs)' in self.feature_groups:
            MA_wins = self._get_win_by_name(mode='MA', 'CSF Log(MA-MA Divs)')
            for feat in feats:
                for i in range(len(MA_wins)-1):
                    df[f'Log(MA_{wins[i]}/MA_{wins[i+1]})_{feat}' ] = np.log(df[feat].rolling(window=wins[i]).mean() / df[feat].rolling(window=wins[i+1]).mean())
                for i in range(len(MA_wins)-2):
                    df[f'Log(MA_{wins[i]}/MA_{wins[i+2]})_{feat}' ] = np.log(df[feat].rolling(window=wins[i]).mean() / df[feat].rolling(window=wins[i+2]).mean())
                for i in range(len(MA_wins)-3):
                    df[f'Log(MA_{wins[i]}/MA_{wins[i+3]})_{feat}' ] = np.log(df[feat].rolling(window=wins[i]).mean() / df[feat].rolling(window=wins[i+3]).mean())
                                       
        if 'CSF (now-MA Diffs)' in self.feature_groups:
            MA_wins = self._get_win_by_name(mode='MA', 'CSF (now-MA Diffs)')
            for feat in feats:
                for w1 in MA_wins:
                    df[f'(now-MA_{w1})_{feat}'] = df[feat] - df[feat].rolling(window=w1).mean()
                                                            
        if 'CSF Log(now-MA Divs)' in self.feature_groups:
            MA_wins = self._get_win_by_name(mode='MA', 'CSF Log(now-MA Divs)')
            for feat in feats:
                for w1 in MA_wins:
                    df[f'Log({feat}/MA_{w1})'] = np.log(df[feat] / df[feat].rolling(window=w1).mean())
                    
        ### Min-Max-Momentum Features
        
        ## Min Features
        if 'CSF (now-Min Diffs)' in self.feature_groups:
            min_max_wins = self._get_win_by_name(mode='MA', 'CSF (now-Min Diffs)')
            for feat in feats:
                for w1 in min_max_wins:
                    df[f'(now-Min_{w1})_{feat}'] = df[feat] - df[feat].rolling(window=w1).min()
                                                             
        if 'CSF TLog(now-Min Diffs)' in self.feature_groups:
            min_max_wins = self._get_win_by_name(mode='MA', 'CSF TLog(now-Min Diffs)')
            for feat in feats:
                for w1 in min_max_wins:
                    df[f'TLog(now-Min_{w1})_{feat}'] = tlog(df[feat] - df[feat].rolling(window=w1).min())
                        
        if 'CSF Log(now-Min Divs)' in self.feature_groups:
            min_max_wins = self._get_win_by_name(mode='MA', 'CSF Log(now-Min Divs)')
            for feat in feats:
                for w1 in min_max_wins:
                    df[f'Log(now/Min_{w1})_{feat}'] = np.log(df[feat] / df[feat].rolling(window=w1).min()) 
                    
        ## Max Features
        if 'CSF (now-Max Diffs)' in self.feature_groups:
            min_max_wins = self._get_win_by_name(mode='MA', 'CSF (now-Max Diffs)')
            for feat in feats:
                for w1 in min_max_wins:
                    df[f'(now-Max_{w1})_{feat}'] = df[feat] - df[feat].rolling(window=w1).max()
                                                             
        if 'CSF TLog(now-Max Diffs)' in self.feature_groups:
            min_max_wins = self._get_win_by_name(mode='MA', 'CSF TLog(now-Max Diffs)')
            for feat in feats:
                for w1 in min_max_wins:
                    df[f'TLog(now-Max_{w1})_{feat}'] = tlog(df[feat] - df[feat].rolling(window=w1).max())
                        
        if 'CSF Log(now-Max Divs)' in self.feature_groups:
            min_max_wins = self._get_win_by_name(mode='MA', 'CSF Log(now-Max Divs)')
            for feat in feats:
                for w1 in min_max_wins:
                    df[f'Log(now/Max_{w1})_{feat}'] = np.log(df[feat] / df[feat].rolling(window=w1).max()) 
        
        """"""""""""""""""""""""""""""""""          
        ### Cumulative Volume Features ###
        """"""""""""""""""""""""""""""""""
        
        if 'CumVol Log' in self.feature_groups: 
            for cvw in cum_volume_windows:
                 df[f'Log(CumVol_({cvw})'] = np.log( df['Volume'].rolling(cvw).sum())
                
        if 'CumVol (now-later Diffs)' in self.feature_groups: 
            for cvw1 in cum_volume_windows[0]:
                for cvw2 in cum_volume_windows[1:]:
                    df[f'({cvw1}-{cvw2}_CumVol)'] =  df['Volume'].rolling(cvw1).sum() - \
                                                     df['Volume'].rolling(cvw2).sum()
             
        if 'CumVol (now-later Divs)' in self.feature_groups: 
            for cvw1 in cum_volume_windows[0]:
                for cvw2 in cum_volume_windows[1:]:
                    df[f'({cvw1}/{cvw2}_CumVol)'] = df['Volume'].rolling(cvw1).sum() / \
                                                    df['Volume'].rolling(cvw2).sum()
                    
        if 'CumVol TLog(now-later Diffs)' in self.feature_groups: 
            for cvw1 in cum_volume_windows[0]:
                for cvw2 in cum_volume_windows[1:]:
                    df[f'({cvw1}-{cvw2}_CumVol)'] = tlog(df['Volume'].rolling(cvw1).sum() - \
                                                         df['Volume'].rolling(cvw2).sum())
                    
        if 'CumVol Log(now-later Divs)' in self.feature_groups:
            for cvw1 in cum_volume_windows[0]:
                for cvw2 in cum_volume_windows[1:]:
                    df[f'Log({cvw1}/{cvw2}_CumVol)'] = np.log(df['Volume'].rolling(cvw1).sum() / \
                                                              df['Volume'].rolling(cvw2).sum())
                    
        """""""""""""""""""""""""""""""""         
        ### STD / Volatility Features ###
        """""""""""""""""""""""""""""""""
        
        if 'STD Close' in self.feature_groups:
            for win in std_windows:
                df[f'STD_{win}_Close'] = df['Close'].rolling(win).std() 
                                                                   
        if 'Log(STD Volume)' in self.feature_groups:
            for win in std_windows:
                df[f'Log(STD_{win})_Volume'] = np.log(df['Volume'].rolling(win).std())
        
        """""""""""""""""""""""""""""""""""""""""
        ### Dropping unused temporal features ###
        """""""""""""""""""""""""""""""""""""""""
    
        if not 'LogVolume' in self.feature_groups:
            self.df.drop(['LogVolume'], axis=1, inplace=True)

        if not ('Candlestick Features' in self.feature_groups): 
            self.df.drop(['Close-High', 'Close-Low', 'High-Low', 'UShadow', 'BShadow'], axis=1, inplace=True)

       
    def _indicator_engineering_scalping(self, df):
        """ This functions contains raw indicator features for scalping.
                Features are mostly suitable for trading decisions and occasionally for time series forecasting.
                
            tsa indicators documentation: https://www.backtrader.com/docu/talibindautoref/   
            ta indicators documentation:  https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html  
        """
        
        MA_windows = [6,15,30]
        
        """""""""""""""""""""""""""
        ###  Trend  Indicators  ###
        """""""""""""""""""""""""""     
        # PSAR not used
        
        if 'ADX-ADXR' in self.extra_indicators:        
            df['ADX']       = talib.ADX( df['High'], df['Low'], df['Close'])
            df['ADXR']      = talib.ADXR(df['High'], df['Low'], df['Close'])
            df['ADX-ADXR']  = df['ADX'] - df['ADXR']
            df.drop(['ADX', 'ADXR'], inplace=True, axis=1)
            
        if 'Aroon' in self.extra_indicators:  
            df['Aroon_Down'], df['Aroon_Up'] = talib.AROON(df['High'], df['Low'])
 
        if 'AMA_Diffs' in self.extra_indicators: 
            df['KAMA']      = talib.KAMA(df['Close'])   
            df['MAMA'], df['FAMA'] = talib.MAMA(df['Close']) 
            df['MAMA-FAMA'] = df['MAMA'] - df['FAMA']
            df['MAMA-KAMA'] = df['MAMA'] - df['KAMA']
            df['FAMA-KAMA'] = df['FAMA'] - df['KAMA']
            df.drop(['KAMA', 'MAMA', 'FAMA'], inplace=True, axis=1)
                       
        if 'CCI' in self.extra_indicators:    
            df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'])
            
        if 'DPO' in self.extra_indicators:  
            cls_DPO  = ta.trend.DPOIndicator(df['Close'])     
            df['DPO'] = cls_DPO.dpo()
            
        if 'EMA' in self.extra_indicators:
            for w in MA_windows:
                df[f'EMA_{w}_(Close)'] = talib.EMA(df['Close'], timeperiod=w)
        
        if 'HMA'in self.extra_indicators:
            for w in MA_windows[1:]:
                df[f'HMA_{w}'] = talib.WMA(talib.WMA(df['Close'], timeperiod=int(w/2))*2 - \
                                           talib.WMA(df['Close'], timeperiod=w), timeperiod=int(np.sqrt(w)))
        if 'KST' in self.extra_indicators:
            cls_KST  = ta.trend.KSTIndicator(df['Close'])
            df['KST']       = cls_KST.kst()
            
        if 'KST_Diff' in self.extra_indicators:    
            cls_KST  = ta.trend.KSTIndicator(df['Close'])
            df['KST_Diff']  = cls_KST.kst_diff()

        if 'MACD_Feats' in self.extra_indicators:    
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close']) 
            
        if 'STC' in self.extra_indicators:
            cls_STC  = ta.trend.STCIndicator(df['Close'])
            df['STC']       = cls_STC.stc() 
            
        if 'VI_Diff' in self.extra_indicators:
            cls_VI   = ta.trend.VortexIndicator(df['High'], df['Low'], df['Close'])
            df['VI_Diff']   = cls_VI.vortex_indicator_diff()      

        if 'WMA'in self.extra_indicators:
            for w in MA_windows:
                df[f'WMA_{w}'] = talib.WMA(df['Close'], timeperiod=w)
            
        """""""""""""""""""""""""""
        ### Momentum Indicators ###
        """""""""""""""""""""""""""
        # MI, TRIX not used   
            
        if 'AO' in self.extra_indicators:
            cls_AO  = ta.momentum.AwesomeOscillatorIndicator(df['High'], df['Low'])
            df['AO']  = cls_AO.awesome_oscillator()
            
        if 'APO' in self.extra_indicators:
            df['APO'] = talib.APO(df['Close'])      
            
        if 'BOP' in self.extra_indicators: 
            df['BOP'] = talib.BOP(df['Open'], df['High'], df['Low'], df['Close'])
            
        if 'CMO' in self.extra_indicators:
            df['CMO'] = talib.CMO(df['Close'])
            
        if 'DX' in self.extra_indicators:
            df['DX']  = talib.DX(df['High'], df['Low'], df['Close'])    
        
        if 'STOCH_Diff' in self.extra_indicators:
            df['Slowk'], df['Slowd'] = talib.STOCH(df['High'], df['Low'], df['Close'])
            df['STOCH_Diff'] = df['Slowk'] - df['Slowd'] 
            df.drop(['Slowk', 'Slowd'], inplace=True, axis=1)
            
        if 'STOCHF_Diff' in self.extra_indicators:
            df['Fastk'], df['Fastd'] = talib.STOCHF(df['High'], df['Low'], df['Close'])
            df['STOCHF_Diff'] = df['Fastk'] - df['Fastd']
            df.drop(['Fastk', 'Fastd'], inplace=True, axis=1)
             
        if 'STOCH_RSI_Diff' in self.extra_indicators:    
            df['Fastk_RSI'], df['Fastd_RSI'] = talib.STOCHRSI(df['Close'])
            df['STOCHRSI_Diff'] = df['Fastk_RSI'] - df['Fastd_RSI']
            df.drop(['Fastk_RSI', 'Fastd_RSI'], inplace=True, axis=1)
              
        if 'PVO' in self.extra_indicators:
            cls_PVO = ta.momentum.PercentageVolumeOscillator(df['Volume'])
            df['PVO'] = cls_PVO.pvo()
            
        if 'RSI' in self.extra_indicators:
            df['RSI'] = talib.RSI(df['Close'])
            
        if 'TSI' in self.extra_indicators:
            cls_TSI = ta.momentum.TSIIndicator(df['Close'])
            df['TSI'] = cls_TSI.tsi()
            
        if 'WILLR' in self.extra_indicators:
            df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'])
                                             
        """""""""""""""""""""""""""
        ###  Volume Indicators  ###
        """""""""""""""""""""""""""
        # AD, OBV, CMF, NVI not used    
          
        if 'EOM' in self.extra_indicators:
            cls_EOM  = ta.volume.EaseOfMovementIndicator(df['High'], df['Low'], df['Volume'])
            df['EOM'] = cls_EOM.ease_of_movement()
            
        if 'MFI' in self.extra_indicators:
            df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'])
            
        if 'VRSI' in self.extra_indicators:
            df['VRSI'] = talib.RSI(df['Volume'])
                                        
        if 'FI' in self.extra_indicators:      
            cls_FI   = ta.volume.ForceIndexIndicator(df['Close'], df['Volume'])                      
            df['FI']  = cls_FI.force_index()
            
        if 'VPT' in self.extra_indicators:
            cls_VPT  = ta.volume.VolumePriceTrendIndicator(df['Close'], df['Volume'])
            df['VPT'] = cls_VPT.volume_price_trend()
            
        if 'VWAP' in self.extra_indicators:
            cls_VWAP = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume'])
            df['VWAP']= cls_VWAP.volume_weighted_average_price()
            
     
        """""""""""""""""""""""""""""
        ### Volatility Indicators ###
        """""""""""""""""""""""""""""
        # ATR, UI not used    
        
        if 'BB_Diff' in self.extra_indicators:
            df['BB_Upper'], _, df['BB_Lower'] = talib.BBANDS(df['Close'])
            df['BB_Diff'] = df['BB_Upper'] - df['BB_Lower'] 
            df.drop(['BB_Upper', 'BB_Lower'], inplace=True, axis=1)
        if 'TRANGE' in self.extra_indicators:
            df['TRANGE'] = talib.TRANGE(df['High'], df['Low'], df['Close'])

        return df
    
    def _alpha101_engineering_scalping(self, df):
        """ Some features could be suitable for forecasting or RL trading.

            paper: 101 Formulaic Alphas in arXiv:1601.00991v3
            repo:  https://github.com/yli188/WorldQuant_alpha101_code/blob/master
        """
        
        alpha_group_idx = [int(g[-3:]) for g in self.worldquant_groups if 'alphas' in g] #['alpha_group001', 'MAE'] --> [1]
        alphas = Alphas_101(df)
        df = pd.concat([df, alphas.get_alpha_groups(alpha_group_idx)], axis=1)
        
        return df
      
class Orderbook_Feature_Engineering():

    
    def __init__(self, feature_groups=[], filtered_features=[]):
        
        self.feature_groups      = feature_groups
        self.filtered_features   = filtered_features

    def full_engineering(self, df):
        
        df = self._base_feature_engineering(df)
        
        filtered_features = set(filtered_features).intersection(df.columns) # can only drop columns existing in df
        df.drop(filtered_features, inplace=True, axis=1)
        
        df.fillna(method='ffill', inplace=True) #filling occasional NaN-values for log features in their columns
        df.dropna(inplace=True)
        
        return df
        
        
    def _base_feature_engineering(self, df): 
        """ It was decided to just use the 5 most recent orderbook entries to save disk space, have a simpler feature engineering,
                and less redundant information for the model that needs to be filtered by the user. 
                Keep in mind that certain features are more for TSF rather than RLT.
        
            As Bid and Ask Prices remain very close to each other, only Diffs (a-b)
                and not Divs Log(a/b) are taken for non-MA features. If using MA features, these overtime differences will
                be large enough to apply Log(a/b). Volumes need a log transform.
            
            RL models may need both Divs and Diffs to decide with relative AND absolute growths.
                Forecasting models could need only diffs (a-b) but may profit from normalized Divs Log(a/b)
                and would be distracted by high fluctuations.
                As TLog(a-b) is very similar in its curve to Log(a/b), so only Log(a/b) is chosen.
            
            There are no momentum features for Bid and Ask Prices as these are mostly captured with 
                long-term Diffs (a-b) or long-term Divs Log(a/b) of Midpoints. These deviate by just 0.01 $ most of the time to the midpoint.
            
            Grads features are features that graduate when moving down/up the order book entries. E.g. Bid to Bid and Ask to Ask differences
                Spreads features are differences between oppsoite entries like Bid and Ask entries.
                Divs and Diffs label features with Differences and Divisions between two different time stamps.
        """
        

        recorded_entries  = 5
        min_max_windows   = [4,8] 
        Diff_windows      = [1,4]
        
        MA_windows       = [ 20,   50,  150, 350, 1200, 3000]   # windows intended for frequencies of 1s - 100ms
        MA_windows_short = [ 20,   50,  150] 
        MA_windows_long  = [350, 1200, 3000]
        min_max_windows  = MA_windows
        
        momentum_shift_vals =       [ 1,   5,  10,   20, 50, 150, 350, 1000] # Features often remain static! | windows intended for frequencies of 1s - 100ms
        momentum_shift_vals_short = [ 1,   5,  10,   20] 
        momentum_shift_vals_long  = [50, 150, 350, 1000] # long term momentum better for RL
        
        STD_windows = [20, 50, 150, 350]
        
        """""""""""""""""""""""""""""""""""
        ### Bid and Ask Price Features  ###
        """""""""""""""""""""""""""""""""""
        
        ### Price Features
        if 'Price Spreads'  in self.feature_groups:        # Bid-Ask Spreads 
            for tup in [(1,1),(2,2),(3,3),(4,4),(5,5),       # Stepwise
                              (2,1),(1,2),(3,1),(1,3)]:      # Crosswise
                df[f'BA (Spread {tup[0]}-{tup[1]})'] = df[f'Bid_{tup[0]}'] - df[f'Ask_{tup[1]}']

            
        if 'Price Grads'  in self.feature_groups:   # Bid-Bid and Ask-Ask Graduations
            for feat in ['Bid', 'Ask']:
                for tup in [(1,2),(2,3),(3,4),(4,5),    # Stepwise
                                  (1,3),(1,4),(1,5)]:   # Graduations to Top 
                    df[f'{feat}-{feat} (Grad {tup[0]}-{tup[1]})'] = df[f'{feat}_{tup[0]}'] - df[f'{feat}_{tup[1]}']
        
        # No Price momentum included here as it is captured with the midpoint features.
        
        """""""""""""""""""""""""""""""""""
        ### Midpoint Price Features ###
        """""""""""""""""""""""""""""""""""
        
        ### Midpoint Features
        if 'Midpoints'  in self.feature_groups:
            for tup in [(1,1),(2,2),(3,3),(4,4),(5,5)]:
                df[f'Midpoint {tup[0]}-{tup[1]}']  = (df[f'Bid_{tup[0]}'] + df[f'Ask_{tup[1]}'])/2
        
        if 'Midpoint Grads'  in self.feature_groups:
            for tup in [(1,2),(2,3),(3,4),(4,5),    # Stepwise Graduations
                              (1,3),(1,4),(1,5)]:   # Graduations to Top
                df[f'MP Grad({tup[0]}-{tup[1]})'] = (df[f'Bid_{tup[0]}'] + df[f'Ask_{tup[0]}'])/2 - \
                                                    (df[f'Bid_{tup[1]}'] + df[f'Ask_{tup[1]}'])/2 

        if 'MP-Price Spreads'  in self.feature_groups:  
            for tup in [(1,1),(2,2),(3,3),(4,4),(5,5)]:  # Midpoint to corresponding Bid/Ask  # Grad of Bid_1 to MP_1 is the same as for Ask_1
                df[f'MP-Price (Spread {tup[0]}-{tup[1]})'] = (df[f'Bid_{tup[0]}'] + df[f'Ask_{tup[0]}'])/2 - df[f'Bid_{tup[1]}'] 
            
            for feat in ['Bid', 'Ask']:   # Midpoint to unrelated Bid/Ask
                for tup in [(1,2), (1,3), (1,4), (1,5)   # Graduations to Top Midpoint
                                   (2,3), (3,4), (4,5)]: # Stepwise Graduations
                    df[f'MP-{feat} (Spread {tup[0]}-{tup[1]})'] = (df[f'Bid_{tup[0]}'] + df[f'Ask_{tup[0]}'])/2 - df[f'{feat}_{tup[1]}']    
          
        ### Midpoint momentum  
        if 'Midpoint Diffs' in self.feature_groups:
            for i in range(2): # only midpoint 1 and 2
                for msv in momentum_shift_vals:
                    df[f'Diff_{msv}_(MP {i}-{i})'] = ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).diff(msv) 
            
        if 'Midpoint Log(Divs)' in self.feature_groups:
            for i in range(2):
                for msv in mom_wins:
                    self.df[f'Log(Div_{msv})_(MP {i}-{i)'] = np.log( ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2) / \
                                                                     ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).shift(msv) )
                                                                     
        ### Midpoint MA features 
        if 'Midpoint MAs' in self.feature_groups:
            for i in range(2): # only midpoint 1 and 2
                for win in MA_windows:
                    df[f'(MA_{win})_(Midpoint {i}-{i})']  = ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=win).mean()

        ### Midpoint MA momentum
        if 'Midpoint (MA-MA Diffs)' in self.feature_groups:      
            wins = MA_windows
            for i in range(2): # only midpoint 1 and 2
                for j in range(len(wins)-1):
                    df[f'MA_{wins[j]}-MA_{wins[j+1]}_(Midpoint {i}-{i})' ] = ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=wins[j]).mean() - \
                                                                             ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=wins[j+1]).mean()
                for j in range(len(wins)-2):
                    df[f'MA_{wins[j]}-MA_{wins[j+2]}_(Midpoint {i}-{i})' ] = ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=wins[j]).mean() - \
                                                                             ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=wins[j+2]).mean()
                for j in range(len(wins)-3):
                    df[f'MA_{wins[j]}-MA_{wins[j+3]}_(Midpoint {i}-{i})' ] = ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=wins[j]).mean() - \
                                                                             ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=wins[j+3]).mean()
                    
        if 'Midpoint (now-MA Diffs)' in self.feature_groups:
            for i in range(2): # only midpoint 1 and 2
                for w1 in MA_windows:
                    df[f'(now-MA_{w1})_(Midpoint {i}-{i})'] = ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2) - \
                                                              ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=w1).mean()
         
        if 'Midpoint Log(MA-MA Divs)' in self.feature_groups:      
            wins = MA_windows
            for i in range(2): # only midpoint 1 and 2
                for j in range(len(wins)-1):
                    df[f'Log(MA_{wins[j]}/MA_{wins[j+1]})_(Midpoint {i}-{i})'] = np.log( ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=wins[j]).mean() / \
                                                                                         ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=wins[j+1]).mean() )
                for j in range(len(wins)-2):
                    df[f'Log(MA_{wins[j]}/MA_{wins[j+2]})_(Midpoint {i}-{i})'] = np.log( ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=wins[j]).mean() / \
                                                                                         ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=wins[j+2]).mean() )
                for j in range(len(wins)-3):
                    df[f'Log(MA_{wins[j]}/MA_{wins[j+3]})_(Midpoint {i}-{i})'] = np.log( ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=wins[j]).mean() / \
                                                                                         ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=wins[j+3]).mean() )
                    
        if 'Midpoint Log(now-MA Divs)' in self.feature_groups:
            MA_windows = self._get_win_by_name(mode='MA', 'HLCV now-MA Spreads')
            for i in range(2):
                for w1 in MA_windows:
                    df[f'Log(now/MA_{w1})_(Midpoint {i}-{i})'] = np.log( ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2) / \
                                                                         ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=w1).mean() )           
           
        ### Midpoint Min-Max momentum                                     
        if 'Midpoint (now-Min|Max Diffs)' in self.feature_groups:
            for i in range(2): # only midpoint 1 and 2
                for w1 in min_max_windows:
                    df[f'(now-Min_{w1})_(Midpoint {i}-{i})'] = ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2) - \
                                                               ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=w1).min()
            for i in range(2): # only midpoint 1 and 2
                for w1 in min_max_windows:
                    df[f'(now-Max_{w1})_(Midpoint {i}-{i})'] = ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2) - \
                                                               ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=w1).max()
                        
        if 'Midpoint Log(now-Min|Max Divs)' in self.feature_groups:
            MA_windows = self._get_win_by_name(mode='MA', 'HLCV now-MA Spreads')
            for i in range(2):
                for w1 in min_max_windows:
                    df[f'Log(now/Min_{w1})_(Midpoint {i}-{i})'] = np.log( ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2) / \
                                                                          ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=w1).min() ) 
            for i in range(2):
                for w1 in min_max_windows:
                    df[f'Log(now/Max_{w1})_(Midpoint {i}-{i})'] = np.log( ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2) / \
                                                                          ((df[f'Bid_{i}'] + df[f'Ask_{i}'])/2).rolling(window=w1).max() ) 

        """""""""""""""""""""""""""""""""""
        ### Bid and Ask Volume Features ###
        """""""""""""""""""""""""""""""""""
        
        ### Log Volumes
        if 'Log Volumes' in self.feature_groups:    # (de-spiked and small-valued)
            for i in range(recorded_entries):
                df[f'Log(AV_{i})'] = np.log(df[f'AskV_{i}']+1)
            for i in range(recorded_entries):
                df[f'Log(BV_{i})'] = np.log(df[f'BidV_{i}']+1)
        
        ### Volume Spreads
        if 'Volume Spreads' in self.feature_groups:   
            for tup in [(1,1),(2,2),(3,3),(4,4),(5,5),  # Stepwise
                              (2,1),(1,2),(3,1),(1,3)]: # Crosswise
                df[f'BidV-AskV (Grad {tup[0]}-{tup[1]})'] = df[f'BidV_1'] - df[f'AskV_2'] 
        
        if 'Volume Spread Divs' in self.feature_groups:   
            for tup in [(1,1),(2,2),(3,3),(4,4),(5,5),  # Stepwise
                              (2,1),(1,2),(3,1),(1,3)]: # Crosswise
                df[f'BidV-AskV Log(Div {tup[0]}-{tup[1]})'] = np.log(df[f'BidV_1'] / df[f'AskV_2'])
                
        if 'Volume TLog Spreads' in self.feature_groups:   
            for tup in [(1,1),(2,2),(3,3),(4,4),(5,5),  # Stepwise
                              (2,1),(1,2),(3,1),(1,3)]: # Crosswise
                df[f'BidV-AskV (Grad {tup[0]}-{tup[1]})'] = tlog(df[f'BidV_1'] - df[f'AskV_2']) 
           
        ### Volume Grads                                
        if 'Volume Grads' in self.feature_groups:    
            for feat in ['Bid', 'Ask']:
                for tup in [(1,2),(2,3),(3,4),(4,5),    # Stepwise
                                  (1,3),(1,4),(1,5)]:   # Graduations to Top 
                    df[f'{feat}V (Grad {tup[0]}-{tup[1]})'] = df[f'{feat}V_1'] - df[f'{feat}V_2'] 
                    
        
        if 'Volume Grad Divs' in self.feature_groups:    
            for feat in ['Bid', 'Ask']:
                for tup in [(1,2),(2,3),(3,4),(4,5),    # Stepwise
                                  (1,3),(1,4),(1,5)]:   # Graduations to Top 
                    df[f'{feat}V Log(Div {tup[0]}-{tup[1]})'] = np.log(df[f'{feat}V_1'] / df[f'{feat}V_2'])
                                                 
        if 'Volume TLog Grads' in self.feature_groups:    
            for feat in ['Bid', 'Ask']:
                for tup in [(1,2),(2,3),(3,4),(4,5),    # Stepwise
                                  (1,3),(1,4),(1,5)]:   # Graduations to Top 
                    df[f'{feat}V TLog(Grad {tup[0]}-{tup[1]})'] = tlog(df[f'{feat}V_1'] - df[f'{feat}V_2']) 
                                                                                                                               
        ## Volume MAs
        if 'Log(Volume MAs)' in self.feature_groups:
            for feat in ['Bid', 'Ask']:
                for i in range(2):
                    for win in MA_windows:
                        df[f'Log(MA_{win}) ({faet}V)'] = np.log( (df[f'{feat}V'] + 1).rolling(window=win).mean() )
                        
        ### Momemtum features are rather considered for Summed Volume Features as there are a lot of them vor each orderbook entry

        """""""""""""""""""""""""""""""""
        ###  Summed Volume Features   ###
        """""""""""""""""""""""""""""""""
        ## For momentum and MA, only Summed Volumes across [1,3,5] values are considered for information density
        
        if 'Log(Summed Vols)' in self.feature_groups:  # Cumulutive sum from Top to bottom over volume entries.
            for feat in ['Bid', 'Ask']: 
                for count in [2,3,4,5]:
                    df[f'{faet}_SVol_{count}'] =  df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1)   
        
        ### Summed Volume MA
        if 'Log(MA SVol)' in self.feature_groups:
            for feat in ['Bid', 'Ask']:
                for count in [1,3,5]:
                    for win in MA_windows:
                        df[f'Log(MA_{win}) ({faet}_SVol_{count})'] = np.log( (df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) ).rolling(window=win).mean())
                        
        ### Summed Volume MA momentum
        if 'SVol TLog(MA-MA Diffs)' in self.feature_groups:      
            wins = MA_windows
            for feat in ['Bid', 'Ask']:
                for count in [1,3,5]:   
                    for j in range(len(wins)-1):
                        df[f'TLog(MA_{wins[j]}-MA_{wins[j+1]})_({faet}_SVol_{count})' ] = 
                                            tlog(( df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) ).rolling(window=wins[j  ]).mean() - \
                                                 ( df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) ).rolling(window=wins[j+1]).mean())
                    for j in range(len(wins)-2):
                        df[f'TLog(MA_{wins[j]}-MA_{wins[j+2]})_({faet}_SVol_{count})' ] = 
                                            tlog(( df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) ).rolling(window=wins[j  ]).mean() - \
                                                 ( df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) ).rolling(window=wins[j+2]).mean())
                    for j in range(len(wins)-3):
                        df[f'TLog(MA_{wins[j]}-MA_{wins[j+3]})_({faet}_SVol_{count})' ] = 
                                            tlog(( df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) ).rolling(window=wins[j  ]).mean() - \
                                                 ( df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) ).rolling(window=wins[j+3]).mean())
                                                 
        if 'SVol Log(MA-MA Divs)' in self.feature_groups:      
            wins = MA_windows
            for feat in ['Bid', 'Ask']:
                for count in [1,3,5]:   
                    for j in range(len(wins)-1):
                        df[f'TLog(MA_{wins[j]}/MA_{wins[j+1]})_({faet}_SVol_{count})' ] = 
                                            np.log(( df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) ).rolling(window=wins[j  ]).mean() / \
                                                   ( df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) ).rolling(window=wins[j+1]).mean())
                    for j in range(len(wins)-2):
                        df[f'TLog(MA_{wins[j]}/MA_{wins[j+2]})_({faet}_SVol_{count})' ] = 
                                            np.log(( df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) ).rolling(window=wins[j  ]).mean() / \
                                                   ( df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) ).rolling(window=wins[j+2]).mean())
                    for j in range(len(wins)-3):
                        df[f'TLog(MA_{wins[j]}/MA_{wins[j+3]})_({faet}_SVol_{count})' ] = 
                                            np.log(( df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) ).rolling(window=wins[j  ]).mean() / \
                                                   ( df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) ).rolling(window=wins[j+3]).mean())
                                        
        if 'SVol TLog(now-MA Diffs)' in self.feature_groups:
            for feat in ['Bid', 'Ask']:
                for count in [1,3,5]: 
                    for w1 in MA_windows:
                        df[f'TLog(now-MA_{w1})_({faet}_SVol_{count})'] = tlog(  df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) - \
                                                                               (df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1)).rolling(window=w1).mean())
         
        if 'SVol Log(now-MA Divs)' in self.feature_groups:
            for feat in ['Bid', 'Ask']:
                for count in [1,3,5]: 
                    for w1 in MA_windows:
                        df[f'Log(now/MA_{w1})_({faet}_SVol_{count})'] = tlog(   df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) / \
                                                                               (df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1)).rolling(window=w1).mean())         
        
        ### Summed Volume Min-Max momentum                                     
        if 'SVol TLog(now-Min|Max Diffs)' in self.feature_groups:
            for feat in ['Bid', 'Ask']:
                for count in [1,3,5]: 
                    for w1 in min_max_windows:
                        df[f'TLog(now-Min_{w1})_({faet}_SVol_{count})'] = tlog(  df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) - \
                                                                                (df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1)).rolling(window=w1).mean())
            for feat in ['Bid', 'Ask']:
                for count in [1,3,5]: 
                    for w1 in min_max_windows:
                        df[f'TLog(now-Max_{w1})_({faet}_SVol_{count})'] = tlog(  df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) - \
                                                                                (df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1)).rolling(window=w1).mean())
                        
        if 'SVol Log(now-Min|Max Divs)' in self.feature_groups:
            for feat in ['Bid', 'Ask']:
                for count in [1,3,5]: 
                    for w1 in min_max_windows:
                        df[f'TLog(now-Min_{w1})_({faet}_SVol_{count})'] = np.log(  df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) / \
                                                                                  (df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1)).rolling(window=w1).mean())
            for feat in ['Bid', 'Ask']:
                for count in [1,3,5]: 
                    for w1 in min_max_windows:
                        df[f'TLog(now-Max_{w1})_({faet}_SVol_{count})'] = np.log(  df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1) / \
                                                                                  (df[[f'{feat}V_{i}' for i in range(1,count+1)]].sum(axis=1)).rolling(window=w1).mean()) 
                                          
        """"""""""""""""""""""""""""""""""
        ### Bid and Ask Worth Features ###
        """"""""""""""""""""""""""""""""""
            
        ### Log Worth
        if 'Log(Worths)'   in self.feature_groups: 
            for feat in ['Bid', 'Ask']:
                for tup in [(1,1),(2,2),(3,3),(4,4),(5,5)]:  # Total (Worth) = Bid*BidV
                    df[f'Log({feat} Worth {tup[0]}-{tup[1]})'] = np.log(df[f'{feat}_{tup[0]}'] * df[f'{feat}V_{tup[1]}'] + 1)
         
        ### Log Spreads
        if 'Worth Log(Spread Divs)' in self.feature_groups:  
            for tup in [(1,1),(2,2),(3,3),(4,4),(5,5),
                              (1,2),(2,1),(1,3),(3,1)]:   
                df[f'BidW-AskW Log(Div {tup[0]}-{tup[1]})'] = np.log((df[f'Bid_{tup[0]}'] * df[f'BidV_{tup[0]}']) / \
                                                                     (df[f'Ask_{tup[1]}'] * df[f'AskV_{tup[1]}']))           
        
        if 'Worth TLog(Spread Diffs)' in self.feature_groups:  
            for tup in [(1,1),(2,2),(3,3),(4,4),(5,5),
                              (1,2),(2,1),(1,3),(3,1)]:   
                df[f'BidW-AskW TLog(Diff {tup[0]}-{tup[1]})'] = tlog((df[f'Bid_{tup[0]}'] * df[f'BidV_{tup[0]}']) - \
                                                                     (df[f'Ask_{tup[1]}'] * df[f'AskV_{tup[1]}']))  
                                                                    
        ### Log Grads  
        if 'Worth Log(Grad Divs)' in self.feature_groups:  
            for feat in ['Bid', 'Ask']:
                for tup in [(1,2),(2,3),(3,4),(4,5),
                                  (1,3),(1,4),(1,5)]:   
                    df[f'{feat}W-{feat}W Log(Div {tup[0]}-{tup[1]})'] = np.log((df[f'{feat}_{tup[0]}'] * df[f'{feat}V_{tup[0]}']) / \
                                                                               (df[f'{feat}_{tup[1]}'] * df[f'{feat}V_{tup[1]}']))           
        
        if 'Worth TLog(Grad Diffs)' in self.feature_groups: 
            for feat in ['Bid', 'Ask']: 
                for tup in [(1,2),(2,3),(3,4),(4,5),
                                  (1,3),(1,4),(1,5)]:   
                    df[f'{feat}W-{feat}W TLog(Diff {tup[0]}-{tup[1]})'] = tlog((df[f'{feat}_{tup[0]}'] * df[f'{feat}V_{tup[0]}']) - \
                                                                               (df[f'{feat}_{tup[1]}'] * df[f'{feat}V_{tup[1]}']))  
        
        STD_windows = [20, 50, 150, 350]
        
        """""""""""""""""""""""""""""""""
        ### STD / Volatility Features ###
        """""""""""""""""""""""""""""""""
        
        if 'STD Midpoint' in self.feature_groups:
            for win in STD_windows:
                for tup in [(1,1)]:
                    df[f'STD_{win}(Midpoint {tup[0]}-{tup[1]})']  = ((df[f'Bid_{tup[0]}'] + df[f'Ask_{tup[1]}'])/2).rolling(win).std()
        
        if 'STD Volumes' in self.feature_groups:
            for feat in ['Bid', 'Ask']:
                for win in STD_windows:
                    for vol_num in [1]:
                        df[f'STD_{win}({feat}V_{vol_num})']  = df[f'{feat}V_{vol_num}'].rolling(win).std()
                                                                                          
        return df

        
        # Then implement here RL Engineering
            

    # def _agent_environment_feature_engineering_hft(self, df): 
        
    #                 ## [action, reward, balance, share_amount,             ### base agent-environment features
    #         ##  bal_now/bal_start, (bal+inv)_now/(bal+inv)_start,          ### similar but easier to learn variation of cumulative return     
    #         ##  Δbalance, Δinvested, Δ(balance+invested), Δshare_amount]   ### absolute changes  
    #     pass  

      
# class Live_RLT_Dataprovider():
#     """ This class exists to manage the fetching behaviour of the class Live_HFT_Sample_Fetcher
#         and provide the most recent timestamp without batching for experimental online inference with online updating
    
#         This class will be used in TradeAI.
            
#         There is no need to extend from the torch Dataset class and no need to use a torch DataLoader as there is no batching
#     """
    
#     #TODO live base engineering
#     # TODO live RL engineering
#     def __init__(self,
#                  live_hft_fetcher : Live_HFT_Sample_Fetcher):
        
#         self.live_hft_fetcher = live_hft_fetcher
#         self.scaler           = scaler           # A predefined scaler with the mean and std parameters of the pretrain_dataset
        
#     def __del__(self):
#         self.live_hft_fetcher.stop_threaded_fetching() # This contructor has to be called by cls.__del__() in order to work and stop the thread.
#                                                        # 'del cls' does not work as it should.
                                                       
#     def update_scaler(self, np_df):  # np_df is like df.to_numpy()   
#         """ The data for updating is kept in a replay buffer """
#         self.scaler = self.scaler.partial_fit(np_df)
        
#     def fetching_start(self):
#         self.live_hft_fetcher.start_threaded_fetching()   
        
#     def get_fetch(self, index): 
#         """ On demand function for getting newest fetch with base feature engineering """
#         if self.live_hft_fetcher.fetch_is_updated:
#             # TODO: Base Feature engineering for every step with Live_Base_Feature_Engineering
#             # TODO: Forecasts of BAV? Maybe later...
#             # TODO: RL   Feature engineering for every step with Live_RL_Feature_Engineering
            
#             # TODO! TODO! : time series normalization against out-of-bound-values?  or real interpreable vals??? --> test
            
#             # TODO: implement a buffer in here (just an unique df of len 10 for data): data, report, agent variables
#             return self.live_hft_fetcher.get_newest_live_fetch()
#         return None
    
#     ## [action, reward, balance, share_amount,                     ### base agent-environment features
#             ##  bal_now/bal_start, (bal+inv)_now/(bal+inv)_start,          ### similar but easier to learn variation of cumulative return     
#             ##  Δbalance, Δinvested, Δ(balance+invested), Δshare_amount]   ### absolute changes 
# class Live_Base_Feature_Engineering():
#     """ Class for doing HFT feature engineering with every new time step.
    
#         More time efficient: uses np.arrays
#     """
#     self.last_live_fetch   = np.hstack(data_arr, 
#                                                np.concatenate(
#                                                    (fetch['bids'][0:5], fetch['asks'][0:5]), axis=None))
    
# class Live_RL_Feature_Engineering():
#     """ Class for doing HFT feature engineering with every new time step
    
#     """
    
    
    
# #TODO: implement different lookback size feature integration in dataprovider

#         #TODO update dropdown names
        
# # TODO: remove either Log Diff/Div or remove Diff/Div
# # TODO cut out flat line and missing data