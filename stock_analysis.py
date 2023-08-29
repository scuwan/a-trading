# 工商银行 601398
import akshare as ak
from pandas import DataFrame
import matplotlib.pyplot as plt

def fetch_hist(code):
    """Return stock history
    >>> fetch_hist('601398').iloc[0]
    日期       2006-10-27
    开盘              3.4
    最高             3.44
    最低             3.26
    收盘             3.28
    成交量        25825396
    成交额    8725367950.0
    振幅             5.77
    涨跌幅            5.13
    涨跌额            0.16
    换手率           37.81
    Name: 0, dtype: object
    """
    hist = ak.stock_zh_a_hist(symbol=code, period='daily', adjust='hfq')
    ohlc = ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
    hist = hist[ohlc]#.iloc[10:] # 剔除股票上市前10天的交易数据
    return hist


def aggregate_stock_hist_sliding_window(stock_hist: DataFrame , window=3, ndays=2, max_num = None):
    dict_features = {}
    dict_labels = {}
    
    iter_num = len(stock_hist) - window - ndays + 1
    if max_num is not None:
        iter_num = min(iter_num, max_num)

    for i in range(iter_num):
        hist_window = stock_hist.iloc[i: i+window]
        hist_window_array = []
        hist_window_array.extend(hist_window['开盘'].to_list())
        hist_window_array.extend(hist_window['最高'].to_list())
        hist_window_array.extend(hist_window['最低'].to_list())
        hist_window_array.extend(hist_window['收盘'].to_list())
        hist_window_array.extend(list(map(lambda x: x/1000000000, hist_window['成交量'].to_list())))
        hist_window_array.extend(list(map(lambda x: x/1000000000, hist_window['成交额'].to_list())))
        hist_window_array.extend(hist_window['振幅'].to_list())
        hist_window_array.extend(hist_window['涨跌幅'].to_list())
        hist_window_array.extend(hist_window['换手率'].to_list())
        dict_features[i] = hist_window_array

        ref_price = hist_window['成交额'].to_list().pop() / hist_window['成交量'].to_list().pop()
        label_window = stock_hist.iloc[i+window: i+window+ndays]
        label_prices = []
        for volume, value in zip(label_window['成交量'].to_list(), label_window['成交额'].to_list()):
            label_prices.append(value/volume)
        dict_labels[i] = [(label_price - ref_price)/ref_price*100 for label_price in label_prices]

    list_features = list(dict_features.values())
    list_labels = list(dict_labels.values())
    return list_features, list_labels


# hist = fetch_hist('601398')
# features, labels = aggregate_stock_hist_sliding_window(hist, ndays= 5, max_num=2)
# print(features)
# print(labels)

def plot_stock_data(code):
    hist = fetch_hist(code)
    hist = hist.reset_index()['收盘']
    plt.plot(hist)
    plt.show()

# # 601398
# plot_stock_data('601398')

if __name__ == "__main__":
    import doctest
    doctest.testmod()
