
from server import get_crypto

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import ta

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error

import tensorflow as tf

dataframes = get_crypto()

for df in dataframes:

    data_df = df
    data_df = data_df.drop(columns=['time','volumefrom'])
    print(data_df)

    data_df = data_df.dropna()

    high = data_df.high
    low = data_df.low
    close = data_df.close

    start = 0
    end = len(data_df)


    x_axis = [i for i in range(start,len(data_df))]


    ichimoku = ta.trend.IchimokuIndicator(high, low)
    span_a = ichimoku.ichimoku_a()
    span_b = ichimoku.ichimoku_b()

    RSI = ta.momentum.RSIIndicator(close)
    rsi = RSI.rsi()

    MACD_indicator = ta.trend.MACD(close)
    MACD = MACD_indicator.macd_diff()

    bollinger_indicator = ta.volatility.BollingerBands(close)
    bollinger_high = bollinger_indicator.bollinger_hband()
    bollinger_low = bollinger_indicator.bollinger_lband()

    wma1 = ta.trend.wma_indicator(close,18)
    wma2 = ta.trend.wma_indicator(close,36)
    hull_inp = (2*(wma1))-wma2
    HULL = ta.trend.wma_indicator(hull_inp,6)

    x_axis = x_axis[start:end]
    close = close[start:end]

    span_a = span_a[start:end]
    span_b = span_b[start:end]

    rsi = rsi[start:end]
    MACD = MACD[start:end]

    data_df = data_df.assign(ichimoku_span_a=span_a)
    data_df = data_df.assign(ichimoku_span_b=span_b)

    data_df = data_df.assign(bollinger_high=bollinger_high)
    data_df = data_df.assign(bollinger_low=bollinger_low)

    data_df = data_df.assign(hull=HULL)

    data_df = data_df.assign(RSI=rsi)
    data_df = data_df.assign(MACD=MACD)

    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)

    #data_df = data_df.drop(columns=['ichimoku_span_a','ichimoku_span_b','bollinger_high','bollinger_low','RSI','MACD'])
    print(data_df)


    #-------------------------------------------------------------------------------------------------------------------------
    #_________________________________________________________________________________________________________________________
    #_________ SPLIT DATASET - TRAIN/TEST ____________________________________________________________________________________



    in_window = 100
    out_window = 30
    test_len = out_window+4


    num_features = len(data_df.columns)
    dataset_len = len(data_df)
    train_len = len(data_df)-test_len


    data = data_df.iloc[:,0:num_features].values
    close = data_df[['close']].values       


    #____________________________________________
    #---------- standard scaling ----------------
    sc = StandardScaler()
    sc2 = StandardScaler()


    train_data = sc.fit_transform(data[:train_len,:])
    train_close = sc2.fit_transform(np.asarray(close[:train_len]).reshape(-1,1))

    #_________________________________________________________________________________________
    #___________Train/Validation Data:________________________________________________________

    x_train = []
    y_train = []

    x_valid = []
    y_valid = []


    for i in range(in_window,len(train_data)-out_window-out_window+1):
        
        x_train.append(train_data[i-in_window:i,:])
        y_train.append(train_close[i:i+out_window,:])
        
        x_valid.append(train_data[i-in_window+out_window:i+out_window,:])
        y_valid.append(train_close[i+out_window:i+out_window+out_window,:])
        
        

    #______reshape train-data to lstm expected shape_______

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],num_features))
    print(x_train.shape,y_train.shape)

    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)

    x_valid = np.reshape(x_valid,(x_valid.shape[0],x_valid.shape[1],num_features))
    print(x_valid.shape,y_valid.shape)

    #_________________________________________________________________________________________
    #______________Test Data:_________________________________________________________________


    #--------- standard-scaling ----------

    test_data = sc.transform(data[train_len-in_window:,:])
    test_close = sc2.transform(np.asarray(close[train_len-in_window:,:]).reshape(-1,1))

    x_test = []
    y_test = []


    for i in range(in_window,len(test_data)-out_window+1):
        
        x_test.append(test_data[i-in_window:i,:])
        y_test.append(test_close[i:i+out_window,:])


    #______reshape test-data to lstm expected shape_______

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    x_test = np.reshape(x_test,( x_test.shape[0], x_test.shape[1],num_features))
    print(x_test.shape, y_test.shape)


    #_________________________________________________________________________________________________________________________

    def build_model(in_window, out_window, num_features):
        
        inputs = tf.keras.layers.Input(shape=(in_window, num_features))
        
        layer = tf.keras.layers.LSTM(64,dropout=0.2, return_sequences=True)(inputs)
        
        layer = tf.keras.layers.LSTM(16)(layer)
        
        outputs = tf.keras.layers.Dense(out_window)(layer)
        
        model =tf.keras.models.Model(inputs, outputs)
        
        
        #opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        #opt = 'Adam'
        opt = 'sgd'
        
        #loss = tf.keras.losses.Huber() 
        loss = 'mse'
        
        model.compile(optimizer=opt, loss=loss, metrics=['mape'])
        
        return model
        
        
        
        
    model_dnn = build_model(in_window, out_window, num_features)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model_dnn.summary()
    hist_simple = model_dnn.fit(x_train, y_train, epochs=200, batch_size=15, callbacks=[callback], shuffle=False, validation_data=(x_valid, y_valid))

    plt.plot(hist_simple.history['loss'])
    plt.plot(hist_simple.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    y_pred = model_dnn.predict(x_test)


    for y, pred in zip(y_test, y_pred):
        
        print("MSE: ", mean_squared_error(y, pred))
        print("MAPE: ", mean_absolute_percentage_error(y, pred))
        print('________________________________')
        
    y_pred = sc2.inverse_transform(y_pred)
    print('\n')


    for i in range(train_len,dataset_len-out_window):
        
        
        width = .89
        width2 = .12

        training_df = data_df.iloc[i-70:i,:]
        up = training_df[training_df.close>= training_df.open]
        down = training_df[training_df.close< training_df.open]

        
        actual_df = data_df.iloc[i:i+out_window,:]
        up_actual = actual_df[actual_df.close >= actual_df.open]
        down_actual = actual_df[actual_df.close < actual_df.open]
        
        
        pred_df = pd.DataFrame(y_pred[i-train_len,:])
        pred_df.columns=['Pred']
        pred_df = pred_df.set_index(actual_df.index)
        
        
        
        print("MAE: ", mean_absolute_error(actual_df.close, y_pred[i-train_len,:]))
        print("MAPE: ", mean_absolute_percentage_error(actual_df.close, y_pred[i-train_len,:]))
        
        
        #plotting predictions 
        ylim_high = actual_df.high.max()
        ylim_low = actual_df.low.min()
        
        ylim_high2 = training_df.high.max()
        ylim_low2 = training_df.low.min()
        
        if ylim_high < ylim_high2:
            ylim_high = ylim_high2
        
        if ylim_low > ylim_low2:
            ylim_low = ylim_low2
            
            
        #define colors to use
        col1 = 'green'
        col2 = 'red'
    
        plt.figure(figsize=(15, 9))
        
        #plot up prices
        plt.bar(up.index, up.close-up.open, width, bottom=up.open, color=col1,)
        plt.bar(up.index, up.high-up.close, width2, bottom=up.close, color=col1)
        plt.bar(up.index, up.low-up.open, width2, bottom=up.open, color=col1)

        #plot down prices
        plt.bar(down.index, down.close-down.open, width, bottom=down.open, color=col2)
        plt.bar(down.index, down.high-down.open, width2, bottom=down.open, color=col2)
        plt.bar(down.index, down.low-down.close, width2, bottom=down.close, color=col2)
        
        plt.bar(up_actual.index, up_actual.close-up_actual.open, width, bottom=up_actual.open, color=col1, alpha=0.6)
        plt.bar(up_actual.index, up_actual.high-up_actual.close, width2, bottom=up_actual.close, color=col1, alpha=0.6)
        plt.bar(up_actual.index, up_actual.low-up_actual.open, width2, bottom=up_actual.open, color=col1, alpha=0.6)

        plt.bar(down_actual.index, down_actual.close-down_actual.open,width, bottom=down_actual.open, color=col2, alpha=0.6)
        plt.bar(down_actual.index, down_actual.high-down_actual.open, width2, bottom=down_actual.open, color=col2, alpha=0.6)
        plt.bar(down_actual.index, down_actual.low-down_actual.close, width2, bottom=down_actual.close, color=col2, alpha=0.6)
        
        
        pred_to_plot = np.insert(pred_df['Pred'].values,0,data_df.iloc[i,3])
        
        plot_start = data_df.loc[i].name
        
        plot_idx = [i for i in range(plot_start, plot_start+out_window+1)]
        
        plt.plot(plot_idx,pred_to_plot,color='purple')
        
        #plt.ylim(ylim_low-10, ylim_high+10)
        
        #rotate x-axis tick labels
        plt.xticks(rotation=45, ha='right')

        #display candlestick chart
        plt.show()

        
        