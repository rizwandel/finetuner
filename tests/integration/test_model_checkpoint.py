import os

import pytest
import torch

import finetuner
from finetuner.tuner.base import BaseTuner
from finetuner.tuner.callback import BestModelCheckpoint, TrainingCheckpoint


@pytest.fixture(scope='module')
def pytorch_model() -> BaseTuner:
    embed_model = torch.nn.Linear(in_features=10, out_features=10)

    return embed_model


def test_both_checkpoints(pytorch_model: BaseTuner, tmpdir, create_easy_data_session):

    data, _ = create_easy_data_session(5, 10, 2)

    finetuner.fit(
        pytorch_model,
        epochs=1,
        train_data=data,
        eval_data=data,
        callbacks=[
            BestModelCheckpoint(save_dir=tmpdir),
            TrainingCheckpoint(save_dir=tmpdir),
        ],
    )

    assert set(os.listdir(tmpdir)) == {'best_model_val_loss', 'saved_model_epoch_01'}
