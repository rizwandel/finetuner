import pytest
import tensorflow as tf
from jina import DocumentArray, DocumentArrayMemmap

from finetuner.tuner.keras import KerasTuner
from finetuner.embedding import set_embeddings
from finetuner.toydata import generate_fashion_match

all_test_losses = [
    'CosineSiameseLoss',
    'CosineTripletLoss',
    'EuclideanSiameseLoss',
    'EuclideanTripletLoss',
]


@pytest.fixture(autouse=True)
def tf_gpu_config():
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=256)]
    )


@pytest.mark.gpu
@pytest.mark.parametrize('loss', all_test_losses)
def test_gpu_keras(generate_random_triplets, loss):
    data = generate_random_triplets(4, 4)
    embed_model = tf.keras.models.Sequential()
    embed_model.add(tf.keras.layers.InputLayer(input_shape=(4,)))
    embed_model.add(tf.keras.layers.Dense(4))

    tuner = KerasTuner(embed_model, loss)

    tuner.fit(data, data, epochs=2, batch_size=4, device='cuda')


@pytest.mark.gpu
def test_set_embeddings_gpu(tmpdir):
    # works for DA
    embed_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32),
        ]
    )
    docs = DocumentArray(generate_fashion_match(num_total=100))
    set_embeddings(docs, embed_model, 'cuda')
    assert docs.embeddings.shape == (100, 32)

    # works for DAM
    dam = DocumentArrayMemmap(tmpdir)
    dam.extend(generate_fashion_match(num_total=42))
    set_embeddings(dam, embed_model, 'cuda')
    assert dam.embeddings.shape == (42, 32)