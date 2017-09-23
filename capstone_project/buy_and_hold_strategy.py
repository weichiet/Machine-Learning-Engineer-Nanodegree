from quantopian.algorithm import calendars

# Called once at the start of the simulation.
def initialize(context):
    # Reference to the SPY security.
    context.spy = sid(8554)
    context.start = True
    schedule_function(market_open, date_rules.every_day(), time_rules.market_open(), calendar=calendars.US_EQUITIES)  
    
def market_open(context, data):
    # Put all my money in SPY.
    if context.start == True:
        order_target_percent(context.spy,1.0)
        context.start = False

 

