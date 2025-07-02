import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, LSTMCell, TimeDistributed, Concatenate, Dropout
from tensorflow.keras.models import Model
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import BayesianOptimization


class LSTMHyperModelTune(kt.HyperModel):
    """
    A HyperModel class for tuning the LSTM-based forecasting model.
    
    Arguments:
        past_steps (int): Number of past time steps used for input.
        forecast_steps (int): Number of future steps to forecast.
        n_cal (int): Number of calendar features.
        n_weather (int): Number of weather features.
        seasonal_lags (int): Number of seasonal lags to use in the model.
    """
    def __init__(self, past_steps: int, forecast_steps: int, n_cal: int, n_weather: int, seasonal_lags: int = 2):
        self.past_steps = past_steps
        self.forecast_steps = forecast_steps
        self.n_cal = n_cal
        self.n_weather = n_weather
        self.P = seasonal_lags

    def build(self, hp):
 
        # Define hyperparameters
        hidden_units = hp.Int("units", 16, 64, step=16)
        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        conv_ratio = hp.Float("conv_ratio", 0.2, 1.0, step=0.1)
        proj_ratio = hp.Float("proj_ratio", 0.2, 0.7, step=0.1)
        encoder_dropout_rate = hp.Choice("encoder_dropout_rate", [0.1, 0.2, 0.3])
        decoder_dropout_rate = hp.Choice("decoder_dropout_rate", [0.1, 0.2, 0.3])
        attention_dropout_rate = hp.Choice("attention_dropout_rate", [0.1, 0.2, 0.3])
        num_heads = hp.Choice("num_heads", [1, 2, 4])

        # Derived variables from hyperparameters
        conv_filters = int(hidden_units * conv_ratio)
        proj_units = int(hidden_units * proj_ratio)

        # Input layers
        past_y_input = Input(shape=(self.past_steps, 1), name="past_load")
        past_x_cal_in = Input(shape=(self.past_steps, self.n_cal), name="past_cal")
        past_x_weather_in = Input(shape=(self.past_steps, self.n_weather), name="past_weather")
        future_x_input = Input(shape=(self.forecast_steps, self.n_cal), name="future_cal")
        future_y_seasonal_in = Input(shape=(self.forecast_steps, self.P), name="future_seasonal_lags")

        # Encoder layers
        conv_past_load = Conv1D(conv_filters, 3, activation='relu', padding='same')(past_y_input)
        conv_past_cal = Conv1D(conv_filters, 3, activation='relu', padding='same')(past_x_cal_in)
        conv_past_weather = Conv1D(conv_filters, 3, activation='relu', padding='same')(past_x_weather_in)

        proj_load_seq = TimeDistributed(Dense(proj_units, activation="relu"))(conv_past_load)
        proj_cal_seq = TimeDistributed(Dense(proj_units, activation="relu"))(conv_past_cal)
        proj_weather_seq = TimeDistributed(Dense(proj_units, activation="relu"))(conv_past_weather)

        encoder_seq = Concatenate(axis=-1)([proj_load_seq, proj_cal_seq, proj_weather_seq])
        encoder_output, state_h, state_c = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True)(encoder_seq)
        encoder_dropout = Dropout(encoder_dropout_rate)(encoder_output)

        # Decoder input
        decoder_input = Concatenate(axis=-1)([future_x_input, future_y_seasonal_in])
        decoder_layer = AttentionDecoder(hidden_units, self.forecast_steps, num_heads=num_heads, attention_dropout_rate=attention_dropout_rate)
        decoder_outputs = decoder_layer(decoder_input, [state_h, state_c], encoder_dropout)
        decoder_dropout = Dropout(decoder_dropout_rate)(decoder_outputs)

        # Forecast output
        forecast_output = TimeDistributed(Dense(1))(decoder_dropout)

        # Model compilation
        model = Model(
            inputs=[past_y_input, past_x_cal_in, past_x_weather_in, future_x_input, future_y_seasonal_in],
            outputs=forecast_output
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
        return model


# Instantiate and run tuner
hypermodel = LSTMHyperModelTune(
    past_steps=past_steps,
    forecast_steps=forecast_steps,
    n_cal=X_tr_cal.shape[2],
    n_weather=X_tr_weather.shape[2],
    seasonal_lags=2
)

# Early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True,
    verbose=2
)

# Initialize Bayesian Optimization tuner
tuner = BayesianOptimization(
    hypermodel=hypermodel,
    objective='val_loss',
    max_trials=15,
    executions_per_trial=2,
    overwrite=True,
)

# Run hyperparameter tuning
tuner.search(
    [X_tr_load, X_tr_cal, X_tr_weather, X_tr_fut_cal, X_tr_fut_seasonal], Y_tr,
    validation_data=([X_val_load, X_val_cal, X_val_weather, X_val_fut_cal, X_val_fut_seasonal], Y_val),
    epochs=500, batch_size=16,
    callbacks=[early_stop],
    verbose=1
)
