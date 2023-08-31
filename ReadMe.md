# 探索实践AI交易模型

|File|Description|
|:----|:----|
|fetch_from_akshare.py| 从[akshare](https://akshare.xyz/)下载A股历史数据|
|lstm_script.py|use [lstm模型](https://arxiv.org/pdf/1402.1128.pdf) to predict the stock price, the model copy from this article - [Stock Prediction and Forecasting Using LSTM(Long-Short-Term-Memory](https://medium.com/@prajjwalchauhan94017/stock-prediction-and-forecasting-using-lstm-long-short-term-memory-9ff56625de73).<br /> Training scripts: _python3 lstm_script.py train_  <br />Predict scripts: _python3 lstm_script.py predict_
|lstm_script_sequence_100_feature_1.py|Some improvements were made to the model implemented in lstm_script.py: the lstm model input features were changed from 100 to 1, and the stock price series used during training was set to 100 by default.
|lstm_script_sequence_100_feature_1_label_1.py|based on lstm_script_sequence_100_feature_1.py, only the last time series output is used to evaluate the model
