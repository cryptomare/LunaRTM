import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.models import Model
from keras import backend as kb
import warnings


def r2(y_true, y_pred):
    ss_res = kb.sum(kb.square(y_true-y_pred))
    ss_tot = kb.sum(kb.square(y_true - kb.mean(y_true)))
    return 1 - ss_res/(ss_tot + kb.epsilon())


class Rover:
    def __init__(
                self,
                input_features=3,
                hidden_units=(128, 64, 32, 8, 4),
                output_features=2,
                droupout=0.01
            ):
        assert isinstance(hidden_units, (tuple, list))
        assert bool(hidden_units)
        assert all([isinstance(i, int) and i > 0 for i in hidden_units])
        assert isinstance(input_features, int) and input_features > 0
        assert isinstance(output_features, int) and output_features > 0

        input_layer = Input(
                shape=(input_features,),
                name='input_layer',
            )
        hidden_layer = Dense(
                units=hidden_units[0],
            )(input_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Activation('relu')(hidden_layer)
        hidden_layer = Dropout(droupout)(hidden_layer)
        for unit in hidden_units[1:]:
            hidden_layer = Dense(
                    units=unit,
                )(hidden_layer)
            hidden_layer = BatchNormalization()(hidden_layer)
            hidden_layer = Activation('relu')(hidden_layer)
            hidden_layer = Dropout(droupout)(hidden_layer)

        output_layer = Dense(
                    units=output_features,
                )(hidden_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation('relu')(output_layer)
        self._model = Model(input_layer, output_layer)
        self._input_features = input_features
        self._output_features = output_features
        self._is_ready = False
        self._is_trained = False

    def ready(self, optimizer='adam', loss='mse', metrics=('mse', r2)):
        self._model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=list(metrics)
            )
        self._is_ready = True

    def learn(
                self,
                x,
                y,
                batch_size=1000,
                epochs=5,
                test_proportion=0.2,
                fold_count=3
            ):
        assert self._is_ready
        assert isinstance(x, np.ndarray) and (np.issubdtype(
            x.dtype, np.floating) or np.issubdtype(x.dtype, np.integer))
        assert isinstance(y, np.ndarray) and (np.issubdtype(
            y.dtype, np.floating) or np.issubdtype(y.dtype, np.integer))
        assert len(x.shape) == 2 and x.shape[1] == self._input_features
        assert len(y.shape) == 2 and y.shape[1] == self._output_features

        x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_proportion, random_state=7
            )
        kfold = KFold(
                n_splits=fold_count, shuffle=True, random_state=23
            )
        for train, validation in kfold.split(x_train, y_train):
            self._model.fit(
                    x=x_train[train],
                    y=y_train[train],
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1
                )
            scores = self._model.evaluate(
                    x_train[validation],
                    y_train[validation],
                    verbose=1,
                    batch_size=1000
                )
            print(scores)
        test_scores = self._model.evaluate(
                x=x_test,
                y=y_test,
                verbose=1
            )
        print('Test Scores\n', test_scores)
        self._is_trained = True
        return 0

    def store(self, file_path):
        if self._is_trained:
            self._model.save(filepath=file_path, include_optimizer=True)
        else:
            warnings.warn('Model is untrained. Model not saved!')
