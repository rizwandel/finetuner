import pytest
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform

from finetuner.tuner.base import BaseLoss
from finetuner.tuner.keras import KerasTuner
from finetuner.tuner.keras.losses import SiameseLoss, TripletLoss


def check_distances(n_cls, vec_embedings, distance):
    # Compute pairwise distances between embeddings
    dists = squareform(pdist(vec_embedings, metric=distance))

    for i in range(n_cls):
        cls_dist = dists[2 * i, 2 * i + 1]
        dist_other = dists[2 * i : 2 * i + 2, :].copy()
        dist_other[:, 2 * i : 2 * i + 2] = 10_000

        assert cls_dist < dist_other.min() - 0.1


@pytest.fixture
def model(dim):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32),
        ]
    )


@pytest.mark.parametrize(
    "n_cls,dim,n_samples,n_epochs,batch_size,loss,distance",
    [
        (5, 10, 100, 10, 25, TripletLoss, 'euclidean'),
        (5, 10, 100, 5, 25, TripletLoss, 'cosine'),
        # Siamese needs more time to converge
        (5, 10, 1000, 5, 256, SiameseLoss, 'euclidean'),
        (5, 10, 1000, 5, 256, SiameseLoss, 'cosine'),
    ],
)
def test_overfit_keras_session(
    create_easy_data_session,
    model,
    n_cls: int,
    dim: int,
    n_samples: int,
    n_epochs: int,
    batch_size: int,
    loss: BaseLoss,
    distance: str,
):
    """This test makes sure that we can overfit the model to a small amount of data.

    We use an over-parametrized model (a few thousand weights for <100 unique input
    vectors), which should easily be able to bring vectors from same class
    together, and put those from different classes apart - note that all the vectors
    are random.
    """

    # Prepare model and data
    data, vecs = create_easy_data_session(n_cls, dim, n_samples)

    # Train
    tuner = KerasTuner(model, loss=loss(distance=distance, margin=0.5))
    tuner.fit(train_data=data, epochs=n_epochs, batch_size=batch_size)

    # Compute embedding for original vectors
    vec_embedings = model(vecs).numpy()

    # Make sure that for each class, the two instances are closer than
    # anything else
    check_distances(n_cls, vec_embedings, distance)


@pytest.mark.parametrize(
    "n_cls,dim,n_epochs,loss,distance",
    [
        (5, 10, 50, TripletLoss, 'euclidean'),
        (5, 10, 100, TripletLoss, 'cosine'),
        # Siamese needs more time to converge
        (5, 10, 100, SiameseLoss, 'euclidean'),
        (5, 10, 100, SiameseLoss, 'cosine'),
    ],
)
def test_overfit_keras_class(
    create_easy_data_class,
    model,
    n_cls: int,
    dim: int,
    n_epochs: int,
    loss: BaseLoss,
    distance: str,
):
    """This test makes sure that we can overfit the model to a small amount of data.

    We use an over-parametrized model (a few thousand weights for <100 unique input
    vectors), which should easily be able to bring vectors from same class
    together, and put those from different classes apart - note that all the vectors
    are random.
    """

    # Prepare model and data
    data, vecs = create_easy_data_class(n_cls, dim)

    # Train
    tuner = KerasTuner(model, loss=loss(distance=distance, margin=0.5))
    tuner.fit(
        train_data=data,
        epochs=n_epochs,
        batch_size=len(data),
        num_items_per_class=2,
        learning_rate=1e-2,  # Found to converge faster here
    )

    # Compute embedding for original vectors
    vec_embedings = model(vecs).numpy()

    # Make sure that for each class, the two instances are closer than
    # anything else
    check_distances(n_cls, vec_embedings, distance)