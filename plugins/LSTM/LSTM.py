from jespipe.plugin.build import Build
from jespipe.plugin.fit import Fit
from jespipe.plugin.predict import Predict
from jespipe.plugin.evaluate import Evaluate
from jespipe.plugin.start import start
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM


class BuildLSTM(Build):
    def __init__(self, parameters):
        """Initialize the build stage of the model.
        
        Keyword Arguments:
        parameters -- parameters passed via command-line necessary for building the model."""
        self.dataset_name = parameters["dataset_name"]
        self.dataframe = parameters["dataframe"]
        self.model_params = parameters["model_params"]

    def build_model(self):
        """Build LSTM RNN model using uncompromised data."""
        sequence_length = self.model_params["sequence_length"]
        feature_count = self.dataframe.shape[1]-1
        learn_rate = self.model_params["learning_rate"]

        # Convert sequence_length and learn_rate to proper data types
        sequence_length = int(sequence_length); learn_rate = float(learn_rate)

        # Split into training and test
        feat_train, label_train, feat_test, label_test = self._load_data(self.dataframe, sequence_length, feature_count)

        # Start building the model using Keras
        model = Sequential()

        for i in range(5):
            model.add(LSTM(feature_count, input_shape=feat_train.shape[1:], return_sequences=True))
            model.add(Dropout(0.1))

        model.add(LSTM(label_train.shape[0], return_sequences=False))
        model.add(Dropout(0.1))
        model.add(Dense(units=label_train.shape[0]))
        model.add(Activation("softmax"))
        opt = Adam(learning_rate=learn_rate)

        # Compile model
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mean_squared_error"])

        # Return created model and training data and testing data
        return model, (feat_train, label_train, feat_test, label_test)

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
        y_train = train[:, -1][:, -1]
        x_test = result[int(row):, :, :-1]
        y_test = result[int(row):, -1][:, -1]

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], feature_count))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], feature_count))

        return x_train, y_train, x_test, y_test


class FitLSTM(Fit):
    def __init__(self, model, feat_train, label_train, parameters):
        """Fit the LSTM model to its training data.
        
        Keyword arguments:
        model -- model to fit to training data.
        feat_train -- data set training features.
        label_train -- data set training labels.
        parameters -- parameters passed via command-line necessary for fitting the model."""
        self.model = model
        self.feat_train = feat_train
        self.label_train = label_train
        self.model_params = parameters["model_params"]
        self.batch_size = self.model_params["batch_size"]
        self.epochs = self.model_params["epochs"]
        self.validation_split = self.model_params["validation_split"]
        self.verbose = self.model_params["verbose"]

    def model_fit(self):
        self.model.fit(
            self.feat_train,
            self.label_train,
            batch_size=int(self.batch_size),
            epochs=int(self.epochs),
            validation_split=float(self.validation_split),
            verbose=int(self.verbose)
        )


class PredictLSTM(Predict):
    def __init__(self, model, predictee):
        """Make predictions on data using the LSTM model.
        
        Keyword arguments:
        model -- model to make predictions with.
        predictee -- data to make prediction on."""
        self.model = model
        self.predictee = predictee

    def model_predict(self):
        prediction = self.model.predict(self.predictee)
        return prediction


class EvaluateLSTM(Evaluate):
    def __init__(self, label_test, model_prediction):
        """Evaluate predictions made by trained LSTM model.
        
        Keyword arguments:
        label_test -- test labels to be used to evaluate model's prediction.
        model_prediction -- LSTM model's prediction to evaulate."""
        self.label_test = label_test
        self.model_prediction = model_prediction

    def model_evaluate(self):
        """Return desired performance metrics"""
        return self._eval_mse, self._eval_rmse

    @property
    def _eval_mse(self):
        """Evaluate the mean squared error of the model's prediction."""
        return mean_squared_error(self.label_test, self.model_prediction)

    @property
    def _eval_rmse(self):
        """Evaluate the root mean squared error of the model's prediction."""
        return np.sqrt(mean_squared_error(self.label_test, self.model_prediction))


if __name__ == "__main__":
    stage, parameters = start()

    # Execute code block based on passed stage from pipeline
    if stage == "train":
        # Pull necessary information from parameters
        dataset_name = parameters["dataset_name"]
        model_name = parameters["model_name"]
        model_save_path = parameters["save_path"]
        model_log_path = parameters["log_path"]
        manip_info = parameters["manip_info"]

        # Normalize data to 0, 1 scale
        sc = MinMaxScaler(feature_range=(0, 1))
        parameters.update({"dataframe": pd.DataFrame(sc.fit_transform(parameters["dataframe"]))})

        # Build the LSTM model
        build_lstm = BuildLSTM(parameters)
        model, data = build_lstm.build_model()

        # Fit the LSTM model on the training data
        fit_lstm = FitLSTM(model, data[0], data[1], parameters)
        fit_lstm.model_fit()
        # TODO: Come up with method to name models
        fit_lstm.model.save(model_save_path, include_optimizer=True)

        # Make a prediction on test set
        predict_lstm = PredictLSTM(fit_lstm.model, data[2])
        prediction = predict_lstm.model_predict()

        # Evaluate model performance on prediction
        evaluate_lstm = EvaluateLSTM(data[3], prediction)
        accuracy = evaluate_lstm.model_evaluate()
        # TODO: Write logging mechanism for accuracy of model

    elif stage == "attack":
        # TODO: Will involve utilizing the load_model function that is a part of the Keras API
        pass

    elif stage == "clean":
        # TODO: Write implementation for the cleaning stage
        pass

    else:
        raise ValueError("Received invalid stage {}. Please only pass valid stages from the pipeline.".format(stage))
