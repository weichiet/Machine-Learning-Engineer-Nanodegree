from quantopian.algorithm import calendars

# Called once at the start of the simulation.
def initialize(context):
    # Reference to the SPY security.
    context.spy = sid(8554)
    
    # Set the slippage and commission values (these are the default settings of Quantopian)
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1))
    
   # Read the benchmark prediction results from Dropbox
    fetch_csv('https://dl.dropboxusercontent.com/s/3bims77nvgism4d/prediction_machine_learning.csv',date_format = '%m-%d-%y')
    context.prediction = symbol('SPY')
    
    # Initialize the variables
    context.current_position = 'CASH'
    context.isShort = False
    context.long_share = 0
    context.short_share = 0
    context.trans_unit = 300

    # Rebalance every day after market open.
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_close(hours=4), calendar=calendars.US_EQUITIES)

    
# This function was scheduled to run once per day 4 hours before maket close
def my_rebalance(context, data):    
    
    if data.current(context.prediction, 'prediction')==1:        
        if context.current_position=='SHORT':
            order(context.spy, context.short_share + context.trans_unit)
            context.current_position = 'LONG'
            context.long_share += context.trans_unit
            context.short_share = 0
        else:
            order(context.spy, context.trans_unit)
            context.current_position = 'LONG'
            context.long_share += context.trans_unit
            
    elif data.current(context.prediction, 'prediction')==0:      
        if context.current_position=='LONG':
            order(context.spy, -1*(context.long_share + context.trans_unit))
            context.current_position = 'SHORT'
            context.short_share += context.trans_unit
            context.long_share = 0
        else:            
            order(context.spy, -1*(context.trans_unit))
            context.current_position = 'SHORT' 
            context.short_share += context.trans_unit      

        
        