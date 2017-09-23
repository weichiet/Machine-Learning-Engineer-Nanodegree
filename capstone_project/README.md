
# Machine Learning Engineer Nanodegree
## Capstone Project: Forecasting Stock Price Movement Direction

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [TensorFlow](https://www.tensorflow.org/)
- [seaborn](https://seaborn.pydata.org/)
- [ploty](https://plot.ly/python/getting-started/)
- [cufflinks](https://plot.ly/ipython-notebooks/cufflinks/)
- [fix-yahoo-finance](https://pypi.python.org/pypi/fix-yahoo-finance)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)


### Code

The `capstone_project.ipynb` notebook illustrates the development process of the project.   
The implemented solution and result are saved as `capstone_project.html`

The following Python files are also needed:  
1) `util.py`: Contains functions for importing input data  
2) `machine_learning_functions.py`: Contains functions for training sklearn machine learning models  
2) `tensor_flow_functions.py`: Contains functions for training multilayer perceptron using TensorFlow   

This project uses a market simulator provided by [Quantopian](https://www.quantopian.com) to backtest a customized trading strategy.  
`proposed_trading_strategy.py` and `buy_and_hold_strategy.py` contain the code to be executed on Quantopian web-based IDE.

### Data

`'/data'` folder contains CSV files of stock indices data used in this project.  
*`Get_Web_Data`* variable in the code controls whether the input data is retrieved from the web or imported from CSV files.  

