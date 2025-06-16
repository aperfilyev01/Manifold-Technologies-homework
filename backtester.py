"""

"""

import talib
from pyalgotrading.constants import *
from pyalgotrading.strategy import StrategyBase
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import sys
import os
import importlib.util
import inspect


# Instrument class. Needed to handle the instrument data.
class Instrument(): 
    def __init__(self, name, path, lot_size=1):
        self.name = name
        self.path = path
        self.lot_size = lot_size

        
    def get_data_1min(self):  # Reads 1-minute date from a given csv file.
        self.data_1min = pd.read_csv(self.path)
        self.data_1min.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        self.data_1min.set_index('date', inplace=True)
        self.data_1min.index = pd.to_datetime(self.data_1min.index)
        
        return

    def resample(self, nmin, toMarket=True):  # Resamples data; most of strategy need 15-min or 60-min intervals; also removes non-market hours
        self.data_resampled = self.data_1min.resample(str(nmin)+'T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'adj_close': 'last',
            'volume': 'sum'
        })
        self.data_resampled['returns'] = (self.data_resampled['adj_close'] / self.data_resampled['adj_close'].shift() - 1.).fillna(0)
        if toMarket:
            self.data_resampled = self.data_resampled.between_time("09:30", "16:00")
            self.data_resampled = self.data_resampled[self.data_resampled.index.dayofweek < 5] 
        self.data_resampled.dropna(inplace=True)
        return
    
    def cutoff(self, ix):   # Cutts off the data to the given timestamp
        self.data = self.data_resampled.loc[:ix]



# Broker class. Only needed to count the total position
class Broker():       
    def __init__(self):
        self.position = {}
    def OrderRegular(self, instrument, action, quantity, order_code=None):
        if action == 'BUY':
            self.position[instrument.name] = quantity
        if action == 'SELL':
            self.position[instrument.name] = -quantity
        return None


# Backtester class. Frovides functions for strategy simulation
class BackTester():
    def __init__(self, Strategy):
        self.broker = Broker()
        self.Strategy = Strategy
        self.pnl = {}
        self.position = {}
        self.execution_shift = 1
        self.log = []
        

    def wrapStrategy(self):  # Creates a wrapping class that implements dummy functions of the class StrategyBase
        class WrappedStrategy(self.Strategy):
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.time_period = None
                self.utils.crossover = self.crossover
                self.initialize()
                
                
            def load_from_yaml(self, path):  # reads strategy parameters from .yams config
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
        
                def flatten_dict(d, parent_key=''):
                    items = {}
                    for k, v in d.items():
                        new_key = f"{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.update(flatten_dict(v, new_key))
                        else:
                            items[new_key.lower()] = v
                    return items
        
                flat_config = flatten_dict(config)
        
                for key, value in flat_config.items():
                    try:
                        setattr(self, key, int(value))
                    except:
                        try:
                            setattr(self, key, float(value))
                        except:
                            setattr(self, key, value)

            def get_historical_data(self, instrument):
                return instrument.data
            
            def crossover(self, val1_hist, val2_hist, accuracy_decimals=2): # crossover function implemented according to its description
                if np.array(val2_hist)[-1] > np.array(val1_hist)[-1] and np.array(val2_hist)[-2] < np.array(val1_hist)[-2] : return 1
                if np.array(val2_hist)[-1] < np.array(val1_hist)[-1] and np.array(val2_hist)[-2] > np.array(val1_hist)[-2] : return -1
                return 0
            
            
        return WrappedStrategy

    def initialize_backtest(self, instrument_bucket, yaml_path): #Initializes a backtest with set of instruments and a yaml config
    
        # Instantiate the strategy
        self.stratW = self.wrapStrategy()()
        
        #load parameters from config
        self.stratW.load_from_yaml(yaml_path)
        
        #define broker
        self.stratW.broker = self.broker
        
        # resample all instruments based on candle_interval parameter
        self.instrument_bucket = instrument_bucket
        for instrument in self.instrument_bucket:
            print(instrument.name, 'resampling', int(self.stratW.candle_interval[:-7]))
            instrument.resample(int(self.stratW.candle_interval[:-7]))
        
        # initialize result variables
        self.pnl = {}
        self.position = {}
        self.pnl_df = None
        self.position_df = None
        self.log = []
        
            
    def step(self, ix): # Chooses the instruments for entry on a given timestamp
        for instrument in self.instrument_bucket:
            instrument.cutoff(ix)
        i_for_entry, meta = self.stratW.strategy_select_instruments_for_entry(None, self.instrument_bucket)
        return i_for_entry, meta
    
    def simulation(self, start_date=None, end_date=None, init_cutoff=100, verbose=True):

        # Define start and end dates if they are None
        if start_date == None:
            start_date = self.instrument_bucket[0].data_resampled.index[0]
        if end_date == None:
            end_date = self.instrument_bucket[0].data_resampled.index[-1]
            
        #for every timestamp...    
        for ix in self.instrument_bucket[0].data_resampled.loc[start_date:end_date].index[init_cutoff:]:

            # Compute PnL based on the current position    
            self.position[ix] = self.broker.position.copy()
            self.pnl[ix] = 0
            for instrument in self.instrument_bucket:
                if instrument.name in self.broker.position.keys():
                    self.pnl[ix] += self.broker.position[instrument.name] * instrument.data_resampled['returns'].shift(-self.execution_shift)[ix] 
            
            # make a step of simulation
            i_for_entry, meta = self.step(ix)
            
            # for every chosen instrument enter the position
            if i_for_entry != []:
                for i in range(len(i_for_entry)):
                    self.stratW.strategy_enter_position(None, i_for_entry[i], meta[i])
                if verbose:
                    print(ix)
                    print([x.name for x in i_for_entry], meta)
                    print('Total Position:', self.broker.position)
                self.log += [str(ix)]
                for x, e in zip(i_for_entry, meta): 
                    self.log += [x.name + ' ' + e['action']]
                self.log += ['Total Position: ' + str(self.broker.position)]
            
        # build TS dataframes form dictionaries    
        self.position_df = pd.DataFrame.from_dict(self.position, orient='index') 
        self.pnl_df = pd.DataFrame.from_dict(self.pnl, orient='index').shift(self.execution_shift) 
           
        return

    def visualize_instrument(self, instrument):  # builds a plot of the strategy's position on the instrument together with instrument's price

        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.plot(instrument.data_resampled['adj_close'].loc[self.position_df.index[0]:self.position_df.index[-1]], color='blue', label='PnL')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        ax2.plot(self.position_df[instrument.name], color='orange', label='Position', alpha=0.7)
        ax2.set_ylabel('Position Size' + ' (' + instrument.name + ')', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Title and grid
        plt.title('Price and Position Over Time')
        fig.tight_layout()
        plt.grid(True)
        
        plt.show()
        
    def visualize_pnl(self):  # builds a plot of the PnL

        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(self.pnl_df.cumsum(), color='blue', label='PnL')
        ax1.set_ylabel('PnL', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        plt.title('PnL Over Time')
        fig.tight_layout()
        plt.grid(True)
        
        plt.show()

    def save_results(self, path): # Saves PnL, position and log to the folder /results
        os.makedirs(path + '/results', exist_ok=True)
        self.pnl_df.to_csv(path + '/results/pnl.csv')
        self.position_df.to_csv(path + '/results/position.csv')
        with open(path + '/results/log.txt', 'w') as f:
            for line in self.log:
                f.write(line + '\n')


def import_single_class_from_file(file_path):  # Imports a strategy class given a folder name
    # Derive module name from filename
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract the class (assuming there's only one top-level class)
    classes = [
        obj for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module_name
    ]

    if len(classes) != 1:
        raise ValueError(f"Expected exactly one class in {file_path}, found {len(classes)}")

    return classes[0]  # Return the class itself, not an instance


# if __name__ == "__main__":
#     aapl = Instrument('AAPL', './data/instruments/AAPL.csv')
#     aapl.get_data_1min()
    
#     path = 'strategy_macd_crossover_delivery'
#     StrategyClass = import_single_class_from_file(path+'/_strategy.py')
#     bt = BackTester(StrategyClass)
#     bt.initialize_backtest([aapl], path+'/_config.yaml')
#     bt.simulation('20240101', None, verbose=False)
#     bt.visualize_instrument(aapl)
#     bt.visualize_pnl()
#     bt.save_results(path)