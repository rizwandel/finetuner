import numpy as np
import pytest
import torch

from finetuner.tuner.pytorch.losses import SiameseLoss, TripletLoss

N_BATCH = 10
N_DIM = 128

ALL_LOSSES = [SiameseLoss, TripletLoss]


@pytest.mark.parametrize('margin', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('distance', ['cosine', 'euclidean'])
@pytest.mark.parametrize('loss_cls', ALL_LOSSES)
def test_loss_output(loss_cls, distance, margin):
    """Test that we get a single positive number as output"""
    loss = loss_cls(distance=distance, margin=margin)

    labels = torch.ones((N_BATCH,))
    labels[: N_BATCH // 2] = 0
    embeddings = torch.rand((N_BATCH, N_DIM))

    output = loss(embeddings, labels)

    assert output.ndim == 0
    assert output >= 0


@pytest.mark.parametrize('distance', ['cosine', 'euclidean'])
@pytest.mark.parametrize('loss_cls', ALL_LOSSES)
def test_loss_zero_same(loss_cls, distance):
    """Sanity check that with perfectly separated embeddings, loss is zero"""

    # Might need to specialize this later
    loss = loss_cls(distance=distance, margin=0.0)

    labels = torch.ones((N_BATCH,))
    labels[: N_BATCH // 2] = 0

    embeddings = torch.ones((N_BATCH, N_DIM))
    embeddings[: N_BATCH // 2] *= -1

    output = loss(embeddings, labels)

    np.testing.assert_almost_equal(output.item(), 0, decimal=5)


@pytest.mark.parametrize(
    'loss_cls,indices,exp_result',
    [
        (SiameseLoss, [[0, 2], [1, 3], [0, 1]], 0.64142),
        (TripletLoss, [[0, 2], [1, 3], [2, 1]], 0.9293),
    ],
)
def test_compute(loss_cls, indices, exp_result):
    """Check that the compute function returns numerically correct results"""

    indices = [torch.tensor(x) for x in indices]
    embeddings = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.4, 0.4], [0.7, 0.7]])
    result = loss_cls(distance='euclidean').compute(embeddings, indices)
    np.testing.assert_almost_equal(result.item(), exp_result, decimal=5)


@pytest.mark.parametrize(
    'loss_cls',
    [SiameseLoss, TripletLoss],
)
def test_compute_loss_given_insufficient_data(loss_cls):
    indices = [torch.tensor([]) for _ in range(3)]
    embeddings = torch.tensor([[0.0, 0.1, 0.2, 0.4]])
    with pytest.raises(ValueError):
        loss_cls(distance='euclidean').compute(embeddings, indices)


@pytest.mark.gpu
@pytest.mark.parametrize(
    'loss_cls',
    [SiameseLoss, TripletLoss],
)
def test_compute_loss_given_insufficient_data_gpu(loss_cls):
    indices = [torch.tensor([]).to('cuda') for _ in range(3)]
    embeddings = torch.tensor([[0.0, 0.1, 0.2, 0.4]]).to('cuda')
    with pytest.raises(ValueError):
        loss_cls(distance='euclidean').compute(embeddings, indices)
