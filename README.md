# SVM Stock Market Prediction Strategy

## Overview

This project implements a trading strategy using a Support Vector Machine (SVM) algorithm to predict stock market trends. The strategy aims to classify future market movements as either upward or downward based on historical data, enabling automated trading decisions.

## Features

- **Historical Data Analysis**: Utilizes past stock data to train the SVM model.
- **Automated Trading**: Executes trades based on model predictions.
- **Risk Management**: Implements rules for profit-taking and stop-loss.

## Strategy Details

### Input Features

The model uses the following features for prediction:
1. Close Price / Mean
2. Current Volume / Average Volume
3. Highest Price / Average Price
4. Lowest Price / Average Price
5. Current Volume
6. Interval Return Rate
7. Interval Standard Deviation

### Trading Rules

- **Entry Rule**: On Mondays, if no positions are held, the model predicts market movement. If an upward movement is predicted, the stock is bought.
- **Exit Rule**: 
  - Positions are sold if the stock price increases by more than 8%.
  - Positions are also sold on Fridays if the weekly gain is less than 1.5%.

## Usage

### Setup

1. **Install Dependencies**:
   ```sh
   pip install scikit-learn
   ```

2. **Run the Strategy**:
   ```python
   from gm.api import *
   import datetime
   import numpy as np
   import pandas as pd
   from sklearn import svm

   # Strategy implementation code goes here
   ```

### Execution

The strategy is executed using the following parameters:

- **Backtest Start Time**: February 1, 2024
- **Backtest End Time**: April 30, 2024
- **Initial Cash**: 88,888,888
- **Commission Ratio**: 0.0002
- **Slippage Ratio**: 0.0001
- **Match Mode**: Match at the close price of the current tick/bar

Run the strategy with:
```python
run(
    strategy_id='7687b30c-4e32-11ef-8fe9-601895265918',
    filename='main1.py',
    mode=MODE_BACKTEST,
    token='7c02d7ddc9acef864899f92db2119c6232d2d697',
    backtest_start_time='2024-02-01 09:00:00',
    backtest_end_time='2024-04-30 09:00:00',
    backtest_adjust=ADJUST_PREV,
    backtest_initial_cash=88888888,
    backtest_commission_ratio=0.0002,
    backtest_slippage_ratio=0.0001,
    backtest_match_mode=1
)
```

### Functions

- **init(context)**: Initializes the strategy, setting parameters like stock symbol, history length, and subscription frequency.
- **on_bar(context, bars)**: Handles incoming bar data, executes trades based on model predictions and trading rules.
- **clf_fit(context, start_date, end_date)**: Trains the SVM model using historical data.
- **on_order_status(context, order)**: Monitors order statuses and prints relevant information.
- **on_backtest_finished(context, indicator)**: Indicates the completion of the backtest.

## Notes

- Ensure that you have the required libraries installed and properly configured.
- Adjust the parameters according to your specific requirements and trading goals.
- Review and understand the strategy before deploying it in a live trading environment to ensure it aligns with your risk tolerance and investment objectives.
- ![image](https://github.com/user-attachments/assets/769e7179-5e91-4b89-bcfb-9463e886fdf5)
