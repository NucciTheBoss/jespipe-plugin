from typing import Tuple

import joblib

import jespipe.plugin.save as save
import numpy as np
import pandas as pd
from jespipe.plugin.start import start
from jespipe.plugin.train.build import Build
from jespipe.plugin.train.evaluate import Evaluate
from jespipe.plugin.train.fit import Fit
from jespipe.plugin.train.predict import Predict
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


class BuildLSTM(Build):
    def __init__(self, parameters: dict) -> None:
        """
        Build class to initialize Sequential LSTM model.
        
        ### Parameters:
        :param parameters: Parameter dictionary sent by Jespipe.

        ### Methods:
        - public
          - build_model (abstract): Build LSTM RNN model using uncompromised data.
        - private
          - _load_data: Internal method for loading/splitting the data into the training and testing data.
        """
        self.dataset_name = parameters["dataset_name"]
        self.dataframe = parameters["dataframe"]
        self.model_params = parameters["model_params"]

    def build_model(self) -> Tuple[Sequential, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Build LSTM RNN model using uncompromised data.

        ### Returns:
        :return: (model, (feat_train, label_train, feat_test, label_test))
        - Positional value of each index in the tuple:
          - 0: An unfitted Sequential LSTM model.
          - 1: The training dataset split into training features, training labels, 
          test features, and test labels.
        """
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

        model.add(LSTM(label_train.shape[0], return_sequences=False))
        model.add(Dropout(0.1))
        model.add(Dense(units=label_train.shape[0]))
        model.add(Activation("softmax"))
        opt = Adam(learning_rate=learn_rate)

        # Compile model
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mean_squared_error"])

        # Return created model and training data and testing data
        return model, (feat_train, label_train, feat_test, label_test)

    def _load_data(self, data: pd.DataFrame, seq_len: int, feature_count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Internal method for loading/splitting the data into the training and testing data.
        
        ### Parameters:
        :param data: Passed dataset to split into training and testing features and labels.
        :param seq_len: User-controlled hyperparameter for LSTM architecture.
        :param feature_count: Number of features in the passed dataset.

        ### Returns:
        :return: (x_train, y_train, x_test, y_test)
        - Positional value of each index in the tuple:
          - 0: Training features.
          - 1: Training labels.
          - 2: Test features.
          - 3: Test labels.
        """
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
    def __init__(self, model: Sequential, feat_train: np.ndarray, 
                    label_train: np.ndarray, parameters: dict) -> None:
        """
        Fit class to facilitate fitting Sequential LSTM model to training data.
        
        ### Paramters:
        :param model: Sequential LSTM model to fit to training data.
        :param feat_train: Training features.
        :param label_train: Training labels.
        :param parameters: Parameter dictionary sent by Jespipe.
        
        ### Methods:
        - public
          - model_fit (abstract): Fit Sequential LSTM model using user-specified hyperparameters.
        """
        self.model = model
        self.feat_train = feat_train
        self.label_train = label_train
        self.model_params = parameters["model_params"]
        self.batch_size = self.model_params["batch_size"]
        self.epochs = self.model_params["epochs"]
        self.validation_split = self.model_params["validation_split"]
        self.verbose = self.model_params["verbose"]

    def model_fit(self) -> None:
        """
        Fit Sequential LSTM model using user-specified hyperparameters.
        """
        self.model.fit(
            self.feat_train,
            self.label_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            verbose=self.verbose
        )


class PredictLSTM(Predict):
    def __init__(self, model: Sequential, predictee: np.ndarray) -> None:
        """
        Prediction class to facilitate making predictions with Sequential LSTM model.
        
        ### Parameters:
        :param model: Sequential LSTM model to make predictions with.
        :param predictee: Data to make prediction on.
        
        ### Methods:
        - public
          - model_predict (abstract): Make prediction on data using Sequential LSTM model.
        """
        self.model = model
        self.predictee = predictee

    def model_predict(self) -> np.ndarray:
        """
        Make prediction on data using Sequential LSTM model.

        ### Returns:
        :return: Sequential LSTM model's prediction
        """
        prediction = self.model.predict(self.predictee)
        return prediction


class EvaluateLSTM(Evaluate):
    def __init__(self, feature_test: np.ndarray, label_test: np.ndarray, 
                    model_to_eval: Sequential) -> None:
        """
        Evaluation class to facilitate evalutions predictions made by fitted Sequential LSTM model.
        
        ### Parameters:
        :param feature_test: Test features.
        :param label_test: Test labels.
        :param model_to_eval: Sequential LSTM model to evaulate.

        ### Methods:
        - public
          - model_evaluate (abstract): Evaluate the mean squared error and root mean squared error of 
          the Sequential LSTM model's prediction.
        - private:
          - _eval_mse: Internal method to evaluate the mean squared error 
          of the Sequential LSTM model's prediction.
          - _eval_rmse: Internal method to evaluate the root 
          mean squared error of the Sequential LSTM model's prediction
        """
        self.feature_test = feature_test
        self.label_test = label_test
        self.model_to_eval = model_to_eval

    def model_evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the mean squared error and root mean squared error of 
        the Sequential LSTM model's prediction.

        ### Returns:
        :return: (mse, rmse)
        - Positional value of each index in the tuple:
          - 0: Mean squared error of model's prediction.
          - 1: Root mean squared error of model's prediction.
        """
        mse = self._eval_mse()
        return mse, self._eval_rmse(mse)

    def _eval_mse(self) -> float:
        """
        Internal method to evaluate the mean squared error 
        of the Sequential LSTM model's prediction.

        ### Returns:
        :return: Mean squared error of the Sequential LSTM model's prediction.
        """
        score = self.model_to_eval.evaluate(self.feature_test, self.label_test, verbose=0)

        # Index 1 is MSE; index 0 is loss
        return score[1]

    def _eval_rmse(self, mse: float) -> float:
        """
        Internal method to evaluate the root mean 
        squared error of the Sequential LSTM model's prediction.
        
        ### Parameters:
        :param mse: Mean squared error of the Sequential LSTM model's prediction.
        
        ### Returns:
        :return: Root mean squared error of the Sequential LSTM model's prediction.
        """
        return np.sqrt(mse)


if __name__ == "__main__":
    stage, parameters = start()

    # Execute code block based on passed stage from pipeline
    if stage == "train":
        # Pull necessary information from parameters
        dataset_name = parameters["dataset_name"]
        model_name = parameters["model_name"]
        dataframe = parameters["dataframe"]
        model_save_path = parameters["save_path"]
        model_log_path = parameters["log_path"]
        model_params = parameters["model_params"]
        manip_params = parameters["manip_params"]
        manip_info = parameters["manip_info"]

        # Normalize data to 0, 1 scale
        sc = MinMaxScaler(feature_range=(0, 1))
        parameters.update({"dataframe": pd.DataFrame(sc.fit_transform(dataframe))})

        # Build the LSTM model
        build_lstm = BuildLSTM(parameters)
        model, data = build_lstm.build_model()

        # Fit the LSTM model on the training data
        fit_lstm = FitLSTM(model, data[0], data[1], parameters)
        fit_lstm.model_fit()

        # Save data to the model_save_path
        save.dictionary(model_save_path, "model_parameters", model_params)
        save.dictionary(model_save_path, "{}_manipulation_parameters".format(manip_info[0]), manip_params)
        save.features(model_save_path, data[2]); save.labels(model_save_path, data[3])
        save.compress_dataframe(model_save_path + "/data", "baseline-data-normalized", dataframe)
        with open(model_save_path + "/model_summary.txt", "wt") as fout: fit_lstm.model.summary(print_fn=lambda x: fout.write(x + "\n"))
        fit_lstm.model.save(model_save_path + "/{}-{}-{}.h5".format(model_name, manip_info[0], manip_info[1]), include_optimizer=True)

        # Make a prediction on test set
        predict_lstm = PredictLSTM(fit_lstm.model, data[2])
        prediction = predict_lstm.model_predict()

        # Save base prediction for later analysis if desired
        save.compress_dataframe(model_save_path + "/data", "baseline-prediction", pd.DataFrame(prediction))

        # Evaluate model performance on prediction
        evaluate_lstm = EvaluateLSTM(data[2], data[3], fit_lstm.model)
        mse, rmse = evaluate_lstm.model_evaluate()
        
        # Create dictionary for logging mse and rmse and then save as a pickle to be loaded back into memory during the attacks
        # 0.0 marks 0.0 pertubation bugdet -> baseline performance
        log_dict = {"0.0": {"mse": mse, "rmse": rmse}}
        save.pickle_object(model_log_path, "mse-rmse", log_dict)

    elif stage == "attack":
        # Load in model to evaluate
        model = parameters["model_path"]; model = load_model(model)

        # Load mse-rmse.pkl file to access dictionary
        log_dict = joblib.load(parameters["log_path"] + "/mse-rmse.pkl")

        # Load test_features
        test_labels = parameters["model_labels"]

        # Loop through each of the adversarial examples
        for adversary in parameters["adver_features"]:
            evaluate_lstm = EvaluateLSTM(joblib.load(adversary), test_labels, model)
            mse, rmse = evaluate_lstm.model_evaluate()
            perturb_budget = adversary.split("/"); perturb_budget = perturb_budget[-1].split("."); perturb_budget = perturb_budget[0]
            log_dict.update({perturb_budget: {"mse": mse, "rmse": rmse}})

        # Once looping through all the adversarial examples has completed, dump updated log dict
        save.pickle_object(parameters["log_path"], "mse-rmse", log_dict)

    else:
        raise ValueError("Received invalid stage {}. Please only pass valid stages from the pipeline.".format(stage))
