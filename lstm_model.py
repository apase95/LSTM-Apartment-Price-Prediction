import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional


class RealEstateLSTM:
    def __init__(self, look_back=12):
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self.n_features = 1
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        self.raw_data = None
        self.split_idx = 0
        self.target_col = 'P_Diff'

    def step1_load_and_preprocess(self, df, test_months=24):
        if df is None:
            print("Không tìm thấy dữ liệu")
            return

        self.raw_data = df
        if self.target_col not in df.columns:
            print(f"Lỗi: Dữ liệu chưa có cột {self.target_col}")
            return

        data_values = df[self.target_col].values.reshape(-1, 1)

        scaled_data = self.scaler.fit_transform(data_values)
        self.split_idx = len(scaled_data) - test_months
        train_set = scaled_data[:self.split_idx]
        self.x_train, self.y_train = self._create_sequences(train_set)
        full_test_input = scaled_data[self.split_idx - self.look_back:]
        self.x_test, self.y_test = self._create_sequences(full_test_input)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], self.n_features)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], self.n_features)

    def _create_sequences(self, dataset):
        X, Y = [], []
        for i in range(len(dataset) - self.look_back):
            X.append(dataset[i:(i + self.look_back), 0])
            Y.append(dataset[i + self.look_back, 0])
        return np.array(X), np.array(Y)

    def step2_build_model(self):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(128, return_sequences=True), #Direction
                                     input_shape=(self.look_back, self.n_features)))
        self.model.add(Dropout(0.2))
        self.model.add(Bidirectional(LSTM(64, return_sequences=False))) #Parameter
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1))  # Output layer
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def step3_train(self, epochs=50, batch_size=16):
        if self.x_train is None:
            return

        self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=1
        )

    def step4_evaluate_and_visualize(self):
        if self.model is None:
            return

        predictions = self.model.predict(self.x_test)
        predictions_real = self.scaler.inverse_transform(predictions)
        y_test_real = self.raw_data[self.target_col].values[self.split_idx:]
        rmse = np.sqrt(mean_squared_error(y_test_real, predictions_real))
        test_dates = self.raw_data.index[self.split_idx:]

        plt.figure(figsize=(14, 10))

        # Biểu đồ Loss
        plt.subplot(2, 1, 1)
        plt.plot(self.history.history['loss'], label='Train Loss', color="red")
        plt.plot(self.history.history['val_loss'], label='Val Loss', color="blue")
        plt.title('Biểu đồ sai số (Loss)')
        plt.legend()

        # Biểu đồ Dự đoán vs Thực tế
        plt.subplot(2, 1, 2)
        plt.plot(self.raw_data.index, self.raw_data[self.target_col],
                 label='Thực tế', color='blue', linewidth=1, alpha=0.7)
        plt.plot(test_dates, predictions_real,
                 label='Dự đoán (Bi-LSTM)', color='red', linewidth=1)
        plt.title(f'So sánh Thực tế và Dự đoán (RMSE: {rmse:.2f})')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá trị Sai phân (Diff)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def step5_predict_future(self):
        if self.model is None:
            return

        last_sequence = self.raw_data[self.target_col].values[-self.look_back:].reshape(-1, 1)
        last_sequence_scaled = self.scaler.transform(last_sequence)
        X_input = last_sequence_scaled.reshape(1, self.look_back, self.n_features)

        pred_scaled = self.model.predict(X_input)
        pred_value = self.scaler.inverse_transform(pred_scaled)

        print(f"Dự báo giá trị {self.target_col} tháng tiếp theo: {pred_value[0][0]:.2f}")

        last_real_price = self.raw_data['Price'].values[-1] if 'Price' in self.raw_data.columns else 0
        predicted_real_price = last_real_price + pred_value[0][0]
        print(f"-> Dự kiến giá thực tế (Price + Diff): {predicted_real_price:.2f} Triệu VND/m2")