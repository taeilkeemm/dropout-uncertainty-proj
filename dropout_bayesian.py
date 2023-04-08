import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, concatenate,LSTM, GRU, Dense, Dropout, Bidirectional, LayerNormalization, MultiHeadAttention, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam


def dataset(label = '역률평균', freq = '1H'):
    # 데이터 읽어들이기
    import json
    with open('Combined_LabelledData_52_{}.json'.format(label), 'rt', encoding='UTF8') as data_file:
        data= json.load(data_file)
    df = pd.DataFrame(data['data'])
    df = df.set_index('TIMESTAMP')
    df.index = pd.to_datetime(df.index)
    
    # 상관관계가 가장 높은 4개를 선정해서 input data로 선정 
    coef = {}
    for i in df['ITEM_NAME'].unique():
        value = np.corrcoef(df[df['ITEM_NAME']==label]['ITEM_VALUE'], df[df['ITEM_NAME']==i]['ITEM_VALUE'])[0,1]
        coef[i] = value
    
    cor_df = pd.DataFrame(list(coef.items()), columns=['ITEM', 'corr'])
    features = cor_df[cor_df['corr'] > 0.9].sort_values(by=['corr'], ascending=False)['ITEM'].unique()
    features = features[:4]
    
    # 선정한 features들을 딥러닝 모델에 넣기 위한 데이터 프레임으로 생성
    idx = df.index.unique()
    ds = pd.DataFrame(index = idx, columns=features)
    for i in features:
        ds[i] = df[df['ITEM_NAME']==i][['ITEM_VALUE']].values

    ds = ds.join(df['LABEL_NAME'])
    ds['LABEL_NAME'] = ds['LABEL_NAME'].map({'정상':0, '비정상':1})

    # 1시간 별로 평균을 구해줘서 1시간 간격 데이터 생성
    ds = ds.groupby(pd.Grouper(freq=freq)).mean()
    ds = ds.astype(np.float32)

    return ds

def preprocess(df, label = '역률평균', timesteps = 24, predict_size = 1):
    # 12시간(timesteps) 전의 데이터를 참고 해 1시간 뒤(predict_size)를 예측하는 dataset을 생성 - train 데이터
    df = df.iloc[:,:-1]
    forecast_idx = df.columns.to_list().index(label)
    split_ratio = 0.75

    split = int(df.shape[0]*split_ratio)
    train_dataset = df[:split]
    test_dataset = df[split:]

    # 정규화
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    scaler = StandardScaler()
    train_scale = scaler.fit_transform(train_dataset) 
    test_scale = scaler.transform(test_dataset)

    def create_dataset(data, window_size, predict_size):
        X_list, y_list = [], []
        predict_size = predict_size - 1
        for i in range(len(data)):
            if (i + window_size+predict_size) < len(data):
                X_list.append(data[i:i+window_size, :])
                y_list.append(data[i+window_size+predict_size, forecast_idx])
        return np.array(X_list), np.array(y_list)


    train_X, train_y = create_dataset(train_scale, timesteps, predict_size)
    test_X, test_y = create_dataset(test_scale, timesteps, predict_size)
    train_y = train_y.reshape(train_y.shape[0],1)
    test_y = test_y.reshape(test_y.shape[0],1)

    return train_X, train_y, test_X, test_y, scaler

class Current():
    def __init__(self, df):
        self.df = df
        self.label = '역률평균'
        self.units = 128 # LSTM 노드 수
        self.dropout_rate = 0.2 # 랜덤으로 Dropout 할 비율
        self.train_X, self.train_y, self.test_X, self.test_y, self.scaler = preprocess(self.df, label=self.label, timesteps=12, predict_size=1)
        self.input_shape = (self.train_X.shape[1],self.train_X.shape[2])
    
    def inverse_transform(self, x):
        # 정규화 했던 것을 원래 값으로 돌려주는 함수
        array = np.concatenate((np.zeros((len(x), self.train_X.shape[2] - 1)), x), axis=1)
        array_transform = self.scaler.inverse_transform(array)[:,-1]

        return array_transform

    def uncertainty_loss(self,true,pred):
        # uncertainty 손실 함수
        mean = pred[:,:self.train_y.shape[-1]]
        var = pred[:,self.train_y.shape[-1]:]
        uncertainty = tf.math.exp(-1*var)
        return 0.5*uncertainty * (true-mean)**2 + 0.5*var

    def LSTM(self):
        inputs = Input(shape=self.input_shape)
        # 양방향 LSTM 2층
        x = Bidirectional(LSTM(self.units,activation='relu',input_shape=self.input_shape , return_sequences=True, dropout=self.dropout_rate, recurrent_dropout = self.dropout_rate))(inputs, training = True)
        x = Bidirectional(LSTM(self.units,activation='relu',input_shape=self.input_shape , dropout=self.dropout_rate, recurrent_dropout = self.dropout_rate))(inputs, training = True)

        # 결과 값은 2개의 Dense 층으로 가는데 한 개는 평균, 한 개는 분산 값 계산을
        mean = Dropout(rate=0.2)(x, training=True)
        mean = Dense(self.train_y.shape[-1])(mean)
        var = Dropout(rate=0.2)(x, training=True)
        var = Dense(self.train_y.shape[-1])(var)
        outputs = concatenate([mean, var])
        model = Model(inputs,outputs)

        model.compile(loss=self.uncertainty_loss, optimizer=Adam())

        return model

    def transformer_encoder(self, inputs, key_dim, num_heads, ff_dim):
        # Normalization and Attention
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, dropout=self.dropout_rate)(x, x)
        x = Dropout(self.dropout_rate)(x, training = True)
        res = x + inputs

        # Feed Forward
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Conv1D(filters=ff_dim, kernel_size=1, activation = 'relu')(x, training = True)
        x = Dropout(0.25)(x, training = True)
        x = Conv1D(filters=inputs.shape[-1], kernel_size=1, activation = 'relu')(x, training = True)

        return x + res    

    def Transfomer_LSTM(self, key_dim = 128, num_heads=2, ff_dim=128, num_blocks=1):
        # 인코더가 LSTM
        inputs = Input(shape=self.input_shape)
        x = Bidirectional(LSTM(self.units, return_sequences=True,activation = 'relu', dropout = self.dropout_rate, recurrent_dropout = self.dropout_rate))(inputs, training = True)
        x = Bidirectional(LSTM(self.units, return_sequences=True,activation = 'relu', dropout = self.dropout_rate, recurrent_dropout = self.dropout_rate))(x, training = True)

        for _ in range(num_blocks):
            x = self.transformer_encoder(x, key_dim, num_heads, ff_dim)

        avg_pool  = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])

        mean = Dropout(rate=0.25)(conc, training=True)
        mean = Dense(self.train_y.shape[-1], activation = 'linear')(mean)
        var = Dropout(rate=0.25)(conc, training=True)
        var = Dense(self.train_y.shape[-1], activation = 'linear')(var)
        outputs = concatenate([mean, var])
        model = Model(inputs,outputs)

        model.compile(loss=self.uncertainty_loss, optimizer=Adam())

        return model

    def Transfomer_Conv(self, key_dim = 256, num_heads=2, ff_dim=128, num_blocks=1):
        # 인코더가 CNN
        inputs = Input(shape=self.input_shape)
        x = Conv1D(filters=self.units, kernel_size=self.train_X.shape[1], strides = 1, activation = 'relu', padding='causal')(inputs, training = True)
        x = Dropout(0.2)(x, training = True)

        for _ in range(num_blocks):
            x = self.transformer_encoder(x, key_dim, num_heads, ff_dim)

        avg_pool  = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        
        mean = Dropout(rate=0.25)(conc, training=True)
        mean = Dense(self.train_y.shape[-1])(mean)
        var = Dropout(rate=0.25)(conc, training=True)
        var = Dense(self.train_y.shape[-1])(var)
        outputs = concatenate([mean, var])
        model = Model(inputs,outputs)

        model.compile(loss=self.uncertainty_loss, optimizer=Adam())
        #model.compile(loss= MSE, optimizer=Adam(), metrics = [MSE])

        return model

    def fit(self, model, epochs = 100, batch_size = 128, show_loss = True):
        # 모델 fitting
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        #early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=50)
    
        history = model.fit(self.train_X, self.train_y, batch_size = batch_size ,epochs=epochs,validation_split=0.2, verbose=1)
        train_loss = model.evaluate(self.train_X, self.train_y)
        print('train loss: {}'.format(train_loss))

        if show_loss:
            plt.figure(figsize=(12,8))
            plt.plot(history.history['loss'], label = 'loss')
            plt.plot(history.history['val_loss'], label = 'val_loss')
            plt.legend()
            plt.title('Model Loss')
            plt.show()
        return model

    def predict_ci(self, model, n_sample, ci=2.58, show= True):
        # 랜덤 샘플링으로 신뢰구간 예측
        from tqdm import tqdm
        eps = np.array([model.predict(self.test_X)[:,:self.test_y.shape[-1]] for _ in tqdm(range(n_sample))])
        ale = np.array([model.predict(self.test_X)[:,self.test_y.shape[-1]:] for _ in tqdm(range(n_sample))])
    
        y_mean = np.mean(eps, axis=0) # predictive mean
        y_var = np.var(eps, axis=0) # epistemic uncertainty
        a_u = np.exp(np.mean(ale, axis=0)) # aleatoric uncertainty
        #v = v.squeeze()
        #a_u = a_u.squeeze()

        test_y_transfrom = self.inverse_transform(self.test_y)
        y_mean_transfrom = self.inverse_transform(y_mean.reshape(-1,1))
        #y_var_transfrom = self.inverse_transform(y_var.reshape(-1,1))

        #99% 신뢰구간
        lower = y_mean - ci*y_var**0.5
        upper = y_mean + ci*y_var**0.5
        lower_transfrom = self.inverse_transform(lower.reshape(-1,1))
        upper_transfrom = self.inverse_transform(upper.reshape(-1,1))

        tf_predict = self.df[-len(self.test_X):]
        tf_predict['predict'] = y_mean_transfrom
        tf_predict['predict_low'] = lower_transfrom
        tf_predict['predict_up'] = upper_transfrom

        under_upper = upper_transfrom[1:] >= test_y_transfrom[:-1]
        over_lower = lower_transfrom[1:] <= test_y_transfrom[:-1]
        total = (under_upper == over_lower)

        # 모델 평가
        from sklearn.metrics import mean_squared_error, r2_score
        print('rmse: {}'.format(np.sqrt(mean_squared_error(self.test_y, y_mean))))
        print('r2: {}'.format(r2_score(self.test_y, y_mean)))
        print("For Uncertainty model, {} are in 99% confidence interval".format(np.mean(total)))
        
        # 데이터의 3시그마 계산
        values = tf_predict[self.label].values
        sigma_upper = np.mean(values) + 3 * np.std(values)
        sigma_lower = np.mean(values) - 3 * np.std(values)
        tf_predict['sigma_upper'] = np.mean(values) + 3 * np.std(values)
        tf_predict['sigma_lower'] = np.mean(values) - 3 * np.std(values)

        # 모델 그래프 결과
        if show == True:
            import matplotlib.pyplot as plt
            tx = self.df.index[-self.test_y.shape[0]:]
            #tx = range(self.test_y.shape[0])

            plt.figure(figsize=(12,8))
            plt.plot(tx, test_y_transfrom,'darkgreen')
            plt.plot(tx, y_mean_transfrom, 'orangered')
            plt.hlines(sigma_upper, tx[0], tx[-1])
            plt.hlines(sigma_lower, tx[0], tx[-1])

            plt.fill_between(tx, y1 = lower_transfrom.reshape(-1), y2= upper_transfrom.reshape(-1), color='lightskyblue')
            plt.xlabel('Time')
            plt.ylabel('Test')
            plt.legend(['Test','Prediction','3sigma'],loc='upper right')
            plt.title('Test Predict')
            plt.show() 
        
        return tf_predict

