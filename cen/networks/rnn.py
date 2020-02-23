#  Copyright 2020 Maruan Al-Shedivat. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  =============================================================================
"""A collection of recurrent models."""

import tensorflow as tf


__all__ = ["BiLSTM", "Transformer"]


def BiLSTM(
    emb_use=True,
    emb_dropout=0.5,
    emb_input_dim=20000,
    emb_output_dim=512,
    emb_mask_zero=False,
    lstm_blocks=1,
    lstm_units=256,
    lstm_activation="tanh",
    lstm_bidirectional=True,
    lstm_post_dropout=0.25,
    lstm_pre_dropout=0.25,
    lstm_recurrent_dropout=0.0,
    lstm_pooling="max",
    lstm_self_attention=False,
):
    """Builds a bidirectional LSTM network with optional global max pooling and
    top dense layers.

    Args:
        emb_use: bool (default: True)
            Whether to use input embeddings.
        emb_dropout: float (default: 0.5)
            Dropout on embeddings.
        emb_input_dim: int (default: 20000)
            Input dim of the embeddings.
        emb_output_dim: int (default: 1024)
            Output dim of the embeddings.
        emb_mask_zero: bool (default: True)
            Makes embedding mask input zeros.

        lstm_blocks: int (default: 1)
            Number of LSTM blocks to stack together.
        lstm_units: int (default: 256)
            Number of units per LSTM layer.
        lstm_activation: str (default: "sigmoid")
            Activation used in the LSTM layers.
        lstm_bidirectional: bool (default: True)
            Makes LSTMs bidirectional.
        lstm_post_dropout: float (default: 0.)
            Dropout on the LSTM outputs.
        lstm_pre_dropout: float (default: 0.25)
            Dropout on the LSTM inputs.
        lstm_recurrent_dropout: float (default: 0.25)
            Dropout on the internal LSTM operations.
        lstm_pooling: str (default: "max")
            Adds global pooling layer.
        lstm_self_attention: bool (default: False)
            Adds multiplicative self-attention layer before pooling.
            Added only if pooling is enabled.

    Returns:
        network: function
            Takes input tensors and builds output tensors.
    """

    def network(inputs):
        previous = inputs
        # Build embeddings.
        if emb_use:
            previous = tf.keras.layers.Embedding(
                input_dim=emb_input_dim,
                output_dim=emb_output_dim,
                mask_zero=emb_mask_zero,
            )(previous)
            previous = tf.keras.layers.Dropout(emb_dropout)(previous)
        # Build the recurrent base.
        for i in range(lstm_blocks):
            Layer = tf.keras.layers.LSTM(
                activation=lstm_activation,
                dropout=lstm_pre_dropout,
                recurrent_dropout=lstm_recurrent_dropout,
                return_sequences=(i + 1 < lstm_blocks) or lstm_pooling,
                units=lstm_units,
            )
            if lstm_bidirectional:
                Layer = tf.keras.layers.Bidirectional(
                    Layer, merge_mode="concat"
                )
            previous = Layer(previous)
            previous = tf.keras.layers.Dropout(lstm_post_dropout)(previous)
        if lstm_pooling and lstm_self_attention:
            previous = tf.keras.layers.Attention()([previous, previous])
        if lstm_pooling == "max":
            previous = tf.keras.layers.GlobalMaxPool1D()(previous)
        elif lstm_pooling == "average":
            previous = tf.keras.layers.GlobalAvgPool1D()(previous)
        return previous

    return network


def Bert(
    pretrained_weights="bert-base-uncased", freeze_weights=True, dropout=0.5
):
    import transformers

    def network(inputs):
        previous = inputs

        # Build transformer layer.
        model = transformers.TFBertModel.from_pretrained(pretrained_weights)

        if freeze_weights:
            for w in model.bert.weights:
                w._trainable = False

        # Return the pooled output.
        pooled_output = model(previous)[1]
        pooled_output = tf.keras.layers.Dropout(dropout)(pooled_output)
        return pooled_output

    return network
