import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def model_data(df):
    print('Preparing data set')
    dates_list = list(df.index)
    scaled_data, scaler = scale_data(df)
    df_scaled = pd.DataFrame(scaled_data,index=(dates_list))
    df_modeled = series_to_supervised(df_scaled, 7, 1)
    return df_modeled, scaler

def scale_data(df):
    values = df.values
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    values=values.reshape(-1, 1) 
    scaled = scaler.fit_transform(values)
    return scaled, scaler

def scale_back_data(df, scaler):
    dates_list = list(df.index)
    new_df = scaler.inverse_transform(df)
    new_df = pd.DataFrame(new_df, index=(dates_list), columns=df.columns)
    return new_df

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    if dropnan: 
        agg.dropna(inplace=True)
    return agg