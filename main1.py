# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
from gm.api import *

import datetime
import numpy as np
import pandas as pd
try:
    from sklearn import svm
except:
    import os
    print('你得先安装scikit-lean啊')
    os.system('pip install scikit-learn')
    from sklearn import svm
    print('scikit-learn库完成嘿嘿')

'''

本策略以SVM算法来训练一个二分类（上涨/下跌）的模型，模型以历史某一段时间数据的涨跌来预测另一段时间的涨跌与否是否有联系。

特征变量为:1.收盘价/均值
          2.现量/均量
          3.最高价/均价
          4.最低价/均价
          5.现量
          6.区间收益率
          7.区间标准差。
若没有仓位，则在每个星期一预测涨跌,并在预测结果为上涨的时候购买标的.
若已经持有仓位，则在盈利大于10%的时候止盈,在星期五涨幅小于2%的时候止盈止损.
'''

def init(context):
    # 选择中国平安的股票
    context.symbol = 'SHSE.000002'
    # 历史窗口长度，N
    context.history_len = 10
    # 预测窗口长度，M
    context.forecast_len = 5
    # 训练样本长度，因为20天为一个交易月
    context.training_len = 60
    # 涨幅大于8%时将股票卖出
    context.earn_rate = 0.08
    # 涨幅小于1.5%时将股票卖出
    context.sell_rate = 0.015
    # 订阅行情
    subscribe(symbols=context.symbol, frequency='60s')


def on_bar(context, bars):
    bar = bars[0]
    # 获取当前时间保存到now
    now = context.now
    now_str = now.strftime('%Y-%m-%d')
    # 获取当前时间的星期保存至weekday
    weekday = now.isoweekday()
    # 历史交易日
    date_list = get_previous_n_trading_dates(exchange='SZSE', date=now_str, n=context.training_len)
    # 上一交易日
    last_date = date_list[-1] 
    last_year_date = date_list[0] 
    # 获取持仓
    position = get_position()

    # 如果当前时间是星期一且没有仓位，则开始预测
    if weekday == 1 and now.hour==9 and now.minute==31 and not position:
        # 获取预测用的历史数据
        features = clf_fit(context,last_year_date,last_date)
        features = np.array(features).reshape(1, -1)
        prediction = context.clf.predict(features)[0]

        # 预测为上涨就买入股票
        if prediction == 1:
            order_target_percent(symbol=context.symbol, percent=1, order_type=OrderType_Limit, position_side=PositionSide_Long, price=bar.close)
            
    # 如果涨幅大于8%,平掉所有仓位止盈
    elif position and bar.close/position[0]['vwap'] >= 1+context.earn_rate:
        order_close_all()

    # 当时间为周五尾盘并且涨幅小于1.5%时,平掉所有仓位止损
    elif position and weekday == 5 and bar.close/position[0]['vwap'] < 1+context.sell_rate and now.hour==14 and now.minute==55:
        order_close_all()


def clf_fit(context,start_date,end_date):
    """
    训练支持向量机模型
    :param start_date:训练样本开始时间
    :param end_date:训练样本结束时间
    """
    # 获取目标股票的daily历史行情
    recent_data = history(context.symbol, frequency='1d', start_time=start_date, end_time=end_date, fill_missing='last',df=True).set_index('eob')
    days = list(recent_data['bob'])

    x_train = []
    y_train = []
    # 整理训练数据
    for index in range(context.history_len, len(recent_data)):
        ## 自变量 X
        # 回溯N个交易日相关数据
        start_date = recent_data.index[index-context.history_len]
        end_date = recent_data.index[index]
        data = recent_data.loc[start_date:end_date,:]
        # 准备训练数据
        close = data['close'].values
        max_x = data['high'].values
        min_n = data['low'].values
        volume = data['volume'].values
        close_mean = close[-1] / np.mean(close)  # 收盘价/均值
        volume_mean = volume[-1] / np.mean(volume)  # 现量/均量
        max_mean = max_x[-1] / np.mean(max_x)  # 最高价/均价
        min_mean = min_n[-1] / np.mean(min_n)  # 最低价/均价
        vol = volume[-1]  # 现量
        return_now = close[-1] / close[0]  # 区间收益率
        std = np.std(np.array(close), axis=0)  # 区间标准差
        # 将计算出的指标添加到训练集X
        x_train.append([close_mean, volume_mean, max_mean, min_mean, vol, return_now, std])
        
        ## 因变量 Y
        if index<len(recent_data)-context.forecast_len:
            y_start_date = recent_data.index[index+1]
            y_end_date = recent_data.index[index+context.forecast_len]
            y_data = recent_data.loc[y_start_date:y_end_date,'close']
            if y_data.iloc[-1] > y_data.iloc[0]:
                label = 1
            else:
                label = 0
            y_train.append(label)
        
        # 最新一期的数据(返回该数据，作为待预测的数据)
        if index==len(recent_data)-1:
            new_x_traain = [close_mean, volume_mean, max_mean, min_mean, vol, return_now, std]
    else:
        # 剔除最后context.forecast_len期的数据
        x_train = x_train[:-context.forecast_len]

    # 训练SVM
    context.clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
                          tol=0.001, cache_size=200, verbose=False, max_iter=-1,decision_function_shape='ovr', random_state=None)
    context.clf.fit(x_train, y_train)

    # 返回最新数据
    return new_x_traain


def on_order_status(context, order):
    # 标的代码
    symbol = order['symbol']
    # 委托价格
    price = order['price']
    # 委托数量
    volume = order['volume']
    # 目标仓位
    target_percent = order['target_percent']
    # 查看下单后的委托状态，等于3代表委托全部成交
    status = order['status']
    # 买卖方向，1为买入，2为卖出
    side = order['side']
    # 开平仓类型，1为开仓，2为平仓
    effect = order['position_effect']
    # 委托类型，1为限价委托，2为市价委托
    order_type = order['order_type']
    if status == 3:
        if effect == 1:
            if side == 1:
                side_effect = '开多仓'
            else:
                side_effect = '开空仓'
        else:
            if side == 1:
                side_effect = '平空仓'
            else:
                side_effect = '平多仓'
        order_type_word = '限价' if order_type==1 else '市价'
        print('{}:标的：{}，操作：以{}{}，委托价格：{}，委托数量：{}'.format(context.now,symbol,order_type_word,side_effect,price,volume))
       

def on_backtest_finished(context, indicator):
    print('*'*50)
    print('回测已完成。')


if __name__ == '__main__':
    '''
    backtest_start_time回测开始时间
    backtest_end_time回测结束时间
    backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
    backtest_initial_cash回测初始资金
    backtest_commission_ratio回测佣金比例
    backtest_slippage_ratio回测滑点比例
    backtest_match_mode市价撮合模式，以下一tick/bar开盘价撮合:0，以当前tick/bar收盘价撮合：1
    '''
    backtest_start_time = str(datetime.datetime.now() - datetime.timedelta(days=180))[:19]
    backtest_end_time = str(datetime.datetime.now())[:19]
    run(strategy_id='7687b30c-4e32-11ef-8fe9-601895265918',
        filename='main1.py',
        mode=MODE_BACKTEST,
        token='7c02d7ddc9acef864899f92db2119c6232d2d697',
        backtest_start_time='2024-02-01 09:00:00',
        backtest_end_time='2024-04-30 09:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=88888888,
        backtest_commission_ratio=0.0002,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)