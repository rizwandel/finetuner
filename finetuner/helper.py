from typing import (
    TypeVar,
    Sequence,
    Iterator,
    Union,
    Callable,
    List,
    Dict,
    Any,
    TYPE_CHECKING,
)

from jina import Document, DocumentArray, DocumentArrayMemmap


if TYPE_CHECKING:
    from .tailor.base import BaseTailor
    from .tuner.base import BaseTuner

AnyDNN = TypeVar(
    'AnyDNN'
)  #: The type of any implementation of a Deep Neural Network object
AnyDataLoader = TypeVar(
    'AnyDataLoader'
)  #: The type of any implementation of a data loader
DocumentSequence = TypeVar(
    'DocumentSequence',
    Sequence[Document],
    DocumentArray,
    DocumentArrayMemmap,
    Iterator[Document],
)  #: The type of any sequence of Document
DocumentArrayLike = Union[
    DocumentSequence,
    Callable[..., DocumentSequence],
]  #: The type :py:data:`DocumentSequence` or a function that gives :py:data:`DocumentSequence`

LayerInfoType = List[
    Dict[str, Any]
]  #: The type of embedding layer information used in Tailor
TunerReturnType = Dict[
    str, Dict[str, Any]
]  #: The type of loss, metric information Tuner returns


def get_framework(dnn_model: AnyDNN) -> str:
    """Return the framework that enpowers a DNN model.

    .. note::
        This is not a solid implementation. It is based on ``__module__`` name,
        the key idea is to tell ``dnn_model`` without actually importing the
        framework.

    :param dnn_model: a DNN model
    :return: `keras`, `torch`, `paddle` or ValueError

    """
    if 'keras' in dnn_model.__module__:
        return 'keras'
    elif 'torch' in dnn_model.__module__:  # note: cover torch and torchvision
        return 'torch'
    elif 'paddle' in dnn_model.__module__:
        return 'paddle'
    else:
        raise ValueError(
            f'can not determine the backend from embed_model from {dnn_model.__module__}'
        )


def get_tuner_class(dnn_model: AnyDNN) -> 'BaseTuner':
    f_type = get_framework(embed_model)

    if f_type == 'keras':
        from .tuner.keras import KerasTuner

        return KerasTuner
    elif f_type == 'torch':
        from .tuner.pytorch import PytorchTuner

        return PytorchTuner
    elif f_type == 'paddle':
        from .tuner.paddle import PaddleTuner

        return PaddleTuner


def get_tailor_class(dnn_model: AnyDNN) -> 'BaseTailor':
    f_type = get_framework(model)

    if f_type == 'keras':
        from .tailor.keras import KerasTailor

        ft = KerasTailor
    elif f_type == 'torch':
        from .tailor.pytorch import PytorchTailor

        ft = PytorchTailor
    elif f_type == 'paddle':
        from .tailor.paddle import PaddleTailor

        ft = PaddleTailor


def is_seq_int(tp) -> bool:
    """Return True if the input is a sequence of integers."""
    return tp and isinstance(tp, Sequence) and all(isinstance(p, int) for p in tp)
