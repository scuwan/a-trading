import akshare as ak

stocks = ak.stock_zh_a_spot_em()
stockcodes = stocks['代码']

# 创建输出文件夹
from datetime import datetime
day = datetime.now().strftime('%Y-%m-%d')
import os
if not os.path.exists(day):
    os.mkdir(day)

ohlc = ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
from tqdm.auto import tqdm
progress_bar = tqdm(total=stockcodes.size)
processed_num = 0
failure_codes = []
for code in stockcodes:
    try:
        stock_history = ak.stock_zh_a_hist(symbol=code, period='daily', adjust='')
        stock_history_qfq = ak.stock_zh_a_hist(symbol=code, period='daily', adjust='qfq')
        stock_history_hfq = ak.stock_zh_a_hist(symbol=code, period='daily', adjust='hfq')
        
        # 对列进行重新排序设置成OHLC(https://en.wikipedia.org/wiki/Open-high-low-close_chart)
        stock_history = stock_history[ohlc]
        stock_history_qfq = stock_history_qfq[ohlc]
        stock_history_hfq = stock_history_hfq[ohlc]

        # 设置以日期为索引
        stock_history.set_index("日期")
        stock_history_qfq.set_index("日期")
        stock_history_hfq.set_index("日期")
        
        # 保存csv
        stock_history.to_csv(f'./{day}/{code}.csv')
        stock_history_qfq.to_csv(f'./{day}/{code}_qfq.csv')
        stock_history_hfq.to_csv(f'./{day}/{code}_hfq.csv')

        # 更新进度条
        progress_bar.update(1)
        processed_num +=1
    except BaseException as e:
        #print(e.args)
        failure_codes.append(code)
        progress_bar.update(1)
        continue

print(f'A 股上市公司总共 {stockcodes.size} 家, 成功处理了 {processed_num} 家')
if len(failure_codes) !=0:
    print(f'处理失败的股票代码:\n{failure_codes}')