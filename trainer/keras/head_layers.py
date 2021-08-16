import abc

import tensorflow as tf
from tensorflow.keras.layers import Layer


class HeadLayer(Layer):
    recommended_loss: str  #: the recommended loss function to be used when equipping this layer to base model

    @abc.abstractmethod
    def call(self, inputs, **kwargs):
        ...


class PairwiseHeadLayer(HeadLayer):
    @abc.abstractmethod
    def call(self, lvalue, rvalue, **kwargs):
        ...


class HatLayer(PairwiseHeadLayer):
    recommended_loss = 'hinge'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(1)

    def call(self, lvalue, rvalue):
        x = tf.concat([lvalue, rvalue, tf.abs(lvalue - rvalue)], axis=-1)
        return self.fc(x)


class DistanceLayer(PairwiseHeadLayer):
    recommended_loss = 'hinge'

    def call(self, lvalue, rvalue):
        return -tf.reduce_sum(
            tf.squared_difference(lvalue, rvalue), axis=-1, keepdims=True
        )


class CosineLayer(PairwiseHeadLayer):
    recommended_loss = 'mse'

    def call(self, lvalue, rvalue):
        normalize_a = tf.nn.l2_normalize(lvalue)
        normalize_b = tf.nn.l2_normalize(rvalue)
        cos_similarity = tf.reduce_sum(
            tf.multiply(normalize_a, normalize_b), axis=-1, keepdims=True
        )
        return cos_similarity
