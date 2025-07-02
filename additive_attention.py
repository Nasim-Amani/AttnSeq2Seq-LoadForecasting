import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, LSTMCell, TimeDistributed, Concatenate, Dropout, Layer
from tensorflow.keras.models import Model


class AdditiveAttention(Layer):
    def __init__(self, units, dropout_rate=0.1):
        super(AdditiveAttention, self).__init__()
        self.units = units

        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
        self.dropout = Dropout(dropout_rate)

    def call(self, encoder_output, decoder_hidden):
        decoder_hidden_expanded = tf.expand_dims(decoder_hidden, 1)  # (batch_size, 1, hidden_dim)

        # Project encoder and decoder hidden states
        projected_encoder = self.W1(encoder_output)  # (batch_size, seq_len, units)
        projected_decoder = self.W2(decoder_hidden_expanded)  # (batch_size, 1, units)

        # Add and apply non-linearity
        additive_score = tf.nn.tanh(projected_encoder + projected_decoder)

        # Final score
        score = self.V(additive_score)  # (batch_size, seq_len, 1)

        # Softmax across sequence length
        attention_weights = tf.nn.softmax(score, axis=1)
        attention_weights = self.dropout(attention_weights)

        # Compute context vector
        context_vector = tf.reduce_sum(attention_weights * encoder_output, axis=1)  # (batch_size, encoder_dim)

        return context_vector


class AttentionDecoder(Layer):
    def __init__(self, hidden_units, forecast_steps, dropout_rate=0.1):
        super(AttentionDecoder, self).__init__()
        self.hidden_units = hidden_units
        self.forecast_steps = forecast_steps
        self.attention = AdditiveAttention(hidden_units)
        self.lstm_cell = LSTMCell(hidden_units)
        self.input_proj = Dense(hidden_units)
        self.output_dense = Dense(1)

    def call(self, inputs, states, encoder_output):
        batch_size = tf.shape(inputs)[0]
        outputs = tf.TensorArray(dtype=tf.float32, size=self.forecast_steps)

        hidden_state, cell_state = states

        def step(t, hidden_state, cell_state, outputs):
            decoder_input_t = inputs[:, t, :]  # shape: (batch_size, input_dim)

            # Attention
            context_vector = self.attention(encoder_output, hidden_state)

            # Combine context and input
            decoder_combined_input = tf.concat([decoder_input_t, context_vector], axis=-1)
            decoder_combined_input = self.input_proj(decoder_combined_input)

            # LSTMCell step
            output, [hidden_state, cell_state] = self.lstm_cell(decoder_combined_input, [hidden_state, cell_state])

            outputs = outputs.write(t, output)

            return t + 1, hidden_state, cell_state, outputs

        t0 = tf.constant(0)
        _, hidden_state, cell_state, outputs = tf.while_loop(
            cond=lambda t, *_: t < self.forecast_steps,
            body=step,
            loop_vars=(t0, hidden_state, cell_state, outputs),
            parallel_iterations=32
        )

        outputs = outputs.stack()  # (forecast_steps, batch, hidden_dim)
        outputs = tf.transpose(outputs, [1, 0, 2])  # (batch, forecast_steps, hidden_dim)
        return outputs


past_y_input = Input(shape=(past_steps, 1), name="past_load")
past_x_cal_in = Input(shape=(past_steps, n_cal), name="past_cal")
past_x_weather_in = Input(shape=(past_steps, n_weather), name="past_weather")
future_x_input = Input(shape=(forecast_steps, n_cal), name="future_cal")
future_y_seasonal_in = Input(shape=(forecast_steps, P), name="future_seasonal_lags")

# Encoder
conv_past_load = Conv1D(conv_filters, 3, activation='relu', padding='same')(past_y_input)
conv_past_cal = Conv1D(conv_filters, 3, activation='relu', padding='same')(past_x_cal_in)
conv_past_weather = Conv1D(conv_filters, 3, activation='relu', padding='same')(past_x_weather_in)

proj_load_seq = TimeDistributed(Dense(proj_units, activation="relu"))(conv_past_load)
proj_cal_seq = TimeDistributed(Dense(proj_units, activation="relu"))(conv_past_cal)
proj_weather_seq = TimeDistributed(Dense(proj_units, activation="relu"))(conv_past_weather)

encoder_seq = Concatenate(axis=-1)([proj_load_seq, proj_cal_seq, proj_weather_seq])
encoder_output, state_h, state_c = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True)(encoder_seq)

# Decoder input
decoder_input = Concatenate(axis=-1)([future_x_input, future_y_seasonal_in])

# Decoder with attention
decoder_layer = AttentionDecoder(hidden_units, forecast_steps)
decoder_outputs = decoder_layer(decoder_input, [state_h, state_c], encoder_output)

# Final dense output
final_output = Dropout(0.25)(decoder_outputs)
forecast_output = TimeDistributed(Dense(1))(final_output)

# Full model
model = Model(
    inputs=[
        past_y_input,
        past_x_cal_in,
        past_x_weather_in,
        future_x_input,
        future_y_seasonal_in
    ],
    outputs=forecast_output
)

model.compile(optimizer="adam", loss="mse")
