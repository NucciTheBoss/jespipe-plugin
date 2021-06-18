from jespipe.plugin.build import Build
from jespipe.plugin.fit import Fit
from jespipe.plugin.predict import Predict
from jespipe.plugin.evaluate import Evaluate
from jespipe.plugin.start import start
from jespipe.plugin.run import run
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.utils.np_utils import to_categorical


class BuildLSTM(Build):
    def __init__(self, parameters):
        self.dataset_name = parameters["dataset_name"]
        self.dataframe = parameters["dataframe"]
        self.model_params = parameters["model_params"]

    def build_model(self):
        """Build LSTM RNN model using uncompromised data."""
        sequence_length = self.model_params["sequence_length"]
        feature_count = self.dataframe.shape[1]-1
        learn_rate = self.model_params["learning_rate"]

        # Split into training and test
        feat_train, label_train, feat_test, label_test = self._load_data(self.dataframe, sequence_length, feature_count)

        # Start building the model using Keras
        model = Sequential()

        for i in range(5):
            model.add(LSTM(feature_count, input_shape=feat_train.shape[1:], return_sequences=True))
            model.add(Dropout(0.1))

        model.add(LSTM(label_train.shape[1], return_sequences=False))
        model.add(Dropout(0.1))
        model.add(Dense(units=label_train.shape[1]))
        model.add(Activation("softmax"))
        opt = Adam(learning_rate=learn_rate)

        # Compile model
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        # Return created model
        return model

    def _load_data(self, data, seq_len, feature_count):
        """Loading/splitting the data into the training and testing data.
        
        Keyword arguments:
        data -- the dataset to split into training and testing data.
        seq_len -- user-controlled hyperparameter for LSTM architecture.
        feature_count -- number of features in the data set."""
        result = np.zeros((len(data) - seq_len, seq_len, feature_count+1))

        # Sequence lengths remain together
        # (i.e, 6 consecutive candles stay together at all times if seq_len=6)
        for index in range(len(data) - seq_len):
            result[index] = data[index: index + seq_len]

        # Shuffling with for reproducable results
        np.random.seed(2020)

        # In-place shuffling for saving space
        np.random.shuffle(result)

        # Amount of data to train on. Train: 85%; Test: 15%
        row = len(result) * 0.85
        train = result[:int(row), :]
        x_train = train[:, :, :-1]
        
        # One-hot encoding
        y_train = to_categorical(train[:, -1][:, -1])
        x_test = result[int(row):, :, :-1]
        y_test = to_categorical(result[int(row):, -1][:, -1])

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], feature_count))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], feature_count))

        return x_train, y_train, x_test, y_test


class FitLSTM(Fit):
    def __init__(self):
        pass

    def model_fit(self):
        pass


class PredictLSTM(Predict):
    def __init__(self):
        pass

    def model_predict(self):
        pass


class EvaluateLSTM(Evaluate):
    def __init__(self):
        pass

    def model_evaluate(self):
        pass


if __name__ == "__main__":
    parameters = start()
    LSTM_build = BuildLSTM(); LSTM_fit = FitLSTM()
    LSTM_predict = PredictLSTM(); LSTM_evaluate = EvaluateLSTM()
    run(parameters=parameters, build=LSTM_build, fit=LSTM_fit, predict=LSTM_predict, evaluate=LSTM_evaluate)
