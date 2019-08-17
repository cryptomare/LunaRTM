import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import warnings


class Rover:
    def __init__(
                self,
                input_features=3,
                hidden_units=(64, 32, 16, 8),
                output_features=4
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
                activation='relu'
            )(input_layer)
        for unit in hidden_units[1:]:
            hidden_layer = Dense(
                    units=unit,
                    activation='relu'
                )(hidden_layer)

        output_layer = Dense(
                    units=output_features,
                    activation='relu'
                )(hidden_layer)
        self._model = Model(input_layer, output_layer)
        self._input_features = input_features
        self._output_features = output_features
        self._is_ready = False
        self._is_trained = False

    def ready(self, optimizer='adam', loss='mse', metrics=('mse',)):
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
                epochs=100,
                shuffle=True,
                validation_split=0.8
            ):
        assert self._is_ready
        assert isinstance(x, np.ndarray) and (np.issubdtype(
            x.dtype, np.floating) or np.issubdtype(x.dtype, np.integer))
        assert isinstance(y, np.ndarray) and (np.issubdtype(
            y.dtype, np.floating) or np.issubdtype(y.dtype, np.integer))
        assert len(x.shape) == 2 and x.shape[1] == self._input_features
        assert len(y.shape) == 2 and y.shape[1] == self._output_features
        self._model.fit(
                x=x,
                y=y,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=shuffle,
                validation_split=validation_split,
                verbose=1
            )
        self._is_trained = True
        return 0

    def store(self, file_path):
        if self._is_trained:
            self._model.save(filepath=file_path, include_optimizer=True)
        else:
            warnings.warn('Model is untrained. Model not saved!')
