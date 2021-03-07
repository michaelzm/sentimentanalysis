# source : https://github.com/xuxingya/tf2crf
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow_addons.text.crf import crf_log_likelihood
import tensorflow as tf
import tensorflow.keras.backend as K


def unpack_data(data):
    if len(data) == 2:
        return data[0], data[1], None
    elif len(data) == 3:
        return data
    else:
        raise TypeError("Expected data to be a tuple of size 2 or 3.")

class CRF(tf.keras.layers.Layer):
    """
    Conditional Random Field layer (tf.keras)
    `CRF` can be used as the last layer in a network (as a classifier). Input shape (features)
    must be equal to the number of classes the CRF can predict (a linear layer is recommended).

    Args:
        chain_initializer: the initialize method for transitions, default orthogonal.

    Input shape:
        nD tensor with shape `(batch_size, sentence length, num_classes)`.

    Output shape:
        in training:
            viterbi_sequence: the predicted sequence tags with shape `(batch_size, sentence length)`
            inputs: the input tensor of the CRF layer with shape `(batch_size, sentence length, num_classes)`
            sequence_lengths: true sequence length of inputs with shape `(batch_size)`
            self.transitions: the internal transition parameters of CRF with shape `(num_classes, num_classes)`
        in predicting:
            viterbi_sequence: the predicted sequence tags with shape `(batch_size, sentence length)`

    Masking
        This layer supports keras masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an embedding layer with the `mask_zero` parameter
        set to `True` or add a Masking Layer before this Layer
    """

    def __init__(self, chain_initializer="orthogonal", **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.chain_initializer = tf.keras.initializers.get(chain_initializer)
        self.transitions = None
        self.supports_masking = False
        self.mask = None
        self.accuracy_fn = tf.keras.metrics.Accuracy()

    def get_config(self):
        config = super(CRF, self).get_config()
        config.update({
            "chain_initializer": "orthogonal"
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3
        units = input_shape[-1]
        self.transitions = self.add_weight(
            name="transitions",
            shape=[units, units],
            initializer=self.chain_initializer,
            trainable = True
        )

    def call(self, inputs, mask=None, training=None):

        if mask is None:
            raw_input_shape = tf.slice(tf.shape(inputs), [0], [2])
 
            mask = tf.ones(raw_input_shape)
        sequence_lengths = K.sum(K.cast(mask, 'int32'), axis=-1)

        viterbi_sequence, _ = tfa.text.crf_decode(
            inputs, self.transitions, sequence_lengths
        )

        return viterbi_sequence, inputs, sequence_lengths, self.transitions
    
class ModelWithCRFLoss(tf.keras.Model):
    """Wrapper around the base model for custom training logic."""

    def __init__(self, base_model, use_dsc=False):
        super().__init__()
        self.base_model = base_model
        self.accuracy_fn = tf.keras.metrics.Accuracy(name='accuracy')

    def call(self, inputs):
        return self.base_model(inputs)

    def compute_loss(self, x, y, sample_weight, training=False):
        y_pred = self(x, training=training)
        viterbi_sequence, potentials, sequence_length, chain_kernel = y_pred
        # we now add the CRF loss:
        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]
        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight
        return viterbi_sequence, sequence_length, tf.reduce_mean(crf_loss)

    def accuracy(self, y_true, y_pred):
        viterbi_sequence, potentials, sequence_length, chain_kernel = y_pred
        sample_weights = tf.sequence_mask(sequence_length, y_true.shape[1])
        return self.accuracy_fn(y_true, viterbi_sequence, sample_weights)

    def train_step(self, data):
        x, y, sample_weight = unpack_data(data)

        with tf.GradientTape() as tape:
            viterbi_sequence, sequence_length, crf_loss = self.compute_loss(
                x, y, sample_weight, training=True
            )
        gradients = tape.gradient(crf_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.accuracy_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))

        return {"crf_loss": crf_loss, 'accuracy': self.accuracy_fn.result()}

    def test_step(self, data):
        x, y, sample_weight = unpack_data(data)
        viterbi_sequence, sequence_length, crf_loss = self.compute_loss(x, y, sample_weight)
        self.accuracy_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))
        return {"crf_loss_val": crf_loss, 'val_accuracy': self.accuracy_fn.result()}