import abc
from typing import TYPE_CHECKING, Callable, Generic, List, Optional, Tuple, Union

from .callback import BaseCallback, ProgressBarCallback
from .dataset import ClassDataset, SessionDataset
from .dataset.helper import batch_document_sequence
from .dataset.samplers import ClassSampler, SessionSampler
from .evaluation import Evaluator
from .miner.base import BaseMiner
from .state import TunerState

from ..embedding import embed
from ..helper import AnyDataLoader, AnyDNN, AnyOptimizer, AnyScheduler, AnyTensor

if TYPE_CHECKING:
    from ..helper import (
        DocumentSequence,
        PreprocFnType,
        CollateFnType,
    )


class BaseLoss(Generic[AnyTensor]):
    distance: str
    miner: Optional[BaseMiner]

    """ Base loss class.

    The subclasses should, in addition to implementing the abstract methods defined
    here, also implement the framework-specific "forward" method, where they
    need to first use the miner to mine indices, and then output the loss by running
    ``compute`` on embeddings and outputs of the miner.
    """

    @abc.abstractmethod
    def compute(
        self,
        embeddings: AnyTensor,
        indices: Tuple[AnyTensor, ...],
    ) -> AnyTensor:
        """Compute the loss using embeddings and indices that the miner outputs"""

    @abc.abstractmethod
    def get_default_miner(self, is_session_dataset: bool) -> BaseMiner:
        """Get the default miner for this loss, given the dataset type"""


class BaseTuner(abc.ABC, Generic[AnyDNN, AnyDataLoader, AnyOptimizer, AnyScheduler]):
    state: TunerState

    def __init__(
        self,
        embed_model: Optional[AnyDNN] = None,
        loss: Union[BaseLoss, str] = 'SiameseLoss',
        configure_optimizer: Optional[
            Callable[[AnyDNN], Union[AnyOptimizer, Tuple[AnyOptimizer, AnyScheduler]]]
        ] = None,
        learning_rate: float = 1e-3,
        scheduler_step: str = 'batch',
        callbacks: Optional[List[BaseCallback]] = None,
        device: str = 'cpu',
        **kwargs,
    ):
        """Create the tuner instance.

        :param embed_model: Model that produces embeddings from inputs
        :param loss: Either the loss object instance, or the name of the loss function.
            Currently available losses are ``SiameseLoss`` and ``TripletLoss``
        :param configure_optimizer: A function that allows you to provide a custom
            optimizer and learning rate. The function should take one input - the
            embedding model, and return either just an optimizer or a tuple of an
            optimizer and a learning rate scheduler.

            For Keras, you should provide the learning rate scheduler directly to
            the optimizer using the `learning_rate` argument in its ``__init__``
            function - and this should be an instance of a subclass of
            ``tf.keras.optimizer.schedulers.LearningRateScheduler`` - and not an
            instance of the callback (``tf.keras.callbacks.LearningRateScheduler``).
        :param learning_rate: Learning rate for the default optimizer. If you
            provide a custom optimizer, this learning rate will not apply.
        :param scheduler_step: At which interval should the learning rate sheduler's
            step function be called. Valid options are "batch" and "epoch".

            For Keras, this option has no effect, as ``LearningRateScheduler`` instances
            are called by the optimizer on each step automatically.
        :param callbacks: A list of callbacks. The progress bar callback
            will be pre-prended to this list.
        :param device: The device to which to move the model. Supported options are
            ``"cpu"`` and ``"cuda"`` (for GPU)
        """
        self._embed_model = embed_model
        self._loss = self._get_loss(loss)
        self._learning_rate = learning_rate
        self._scheduler_step = scheduler_step
        self._scheduler = None
        self._device_name = device

        # Check for early stopping
        self.stop_training = False

        # Place model on device
        self._move_model_to_device()

        # Create optimizer (and scheduler)
        if configure_optimizer:
            res = configure_optimizer(self._embed_model)
            if isinstance(res, tuple):
                self._optimizer, self._scheduler = res
            else:
                self._optimizer = res
        else:
            self._optimizer = self._default_configure_optimizer(self._embed_model)

        # Prepare callbacks
        callbacks = callbacks or []
        self._callbacks = [ProgressBarCallback()] + callbacks

    @staticmethod
    def _get_batch_sampler(
        dataset: Union[ClassDataset, SessionDataset],
        batch_size: int,
        shuffle: bool,
        num_items_per_class: Optional[int] = None,
    ) -> Union[ClassSampler, SessionSampler]:
        """Get the batch sampler"""

        if isinstance(dataset, ClassDataset):
            batch_sampler = ClassSampler(
                dataset.labels, batch_size, num_items_per_class
            )
        elif isinstance(dataset, SessionDataset):
            batch_sampler = SessionSampler(dataset.labels, batch_size, shuffle)
        else:
            raise TypeError(
                f'`dataset` must be either {type(SessionDataset)} or'
                f' {type(ClassDataset)}, but receiving {type(dataset)}'
            )

        return batch_sampler

    @abc.abstractmethod
    def _move_model_to_device(self):
        """Move the model to device and set device"""

    @abc.abstractmethod
    def _default_configure_optimizer(self, model: AnyDNN) -> AnyOptimizer:
        """Get the default optimizer (Adam), if none was provided by user."""

    def _trigger_callbacks(self, method: str, **kwargs):
        """Trigger the specified method on all callbacks"""
        for callback in self._callbacks:
            getattr(callback, method)(self, **kwargs)

    @property
    def embed_model(self) -> AnyDNN:
        """Get the base model of this object."""
        return self._embed_model

    def fit(
        self,
        train_data: 'DocumentSequence',
        query_data: Optional['DocumentSequence'] = None,
        index_data: Optional['DocumentSequence'] = None,
        epochs: int = 10,
        batch_size: int = 256,
        num_items_per_class: Optional[int] = None,
        preprocess_fn: Optional['PreprocFnType'] = None,
        collate_fn: Optional['CollateFnType'] = None,
        num_workers: int = 0,
        limit: int = 20,
        distance: str = 'cosine',
        **kwargs,
    ):
        """Finetune the model on the training data.

        :param train_data: Data on which to train the model.
        :param query_data: Search data used by the evaluator at the end of each epoch, to evaluate the model
        :param index_data: Index data or catalog used by the evaluator at the end of each epoch, to evaluate the model
        :param epochs: Number of epochs to train the model.
        :param batch_size: The batch size to use for training and evaluation.
        :param num_items_per_class: Number of items from a single class to include in
            the batch. Only relevant for class datasets.
        :param preprocess_fn: A pre-processing function. It should take as input the
            content of an item in the dataset and return the pre-processed content.
        :param collate_fn: The collation function to merge the content of individual
            items into a batch. Should accept a list with the content of each item,
            and output a tensor (or a list/dict of tensors) that feed directly into the
            embedding model.
        :param num_workers: Number of workers used for loading the data.
            This works only with Pytorch and Paddle Paddle, and has no effect when using
            a Keras model.
        :param limit: The number of top search results to consider, when evaluating.
        :param distance: The type of distance metric to use when matching query and index docs during evaluation,
            available options are ``"cosine"``, ``"euclidean"`` and ``"sqeuclidean"``.
        """
        ...

        try:
            self._fit(
                train_data,
                eval_data,
                epochs,
                batch_size,
                num_items_per_class,
                preprocess_fn,
                collate_fn,
                num_workers,
            )
        except KeyboardInterrupt:
            self._trigger_callbacks('on_keyboard_interrupt')
        except BaseException as e:
            self._trigger_callbacks('on_exception', exception=e)
            raise

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        """Save the weights of the :py:attr:`.embed_model`."""
        ...

    @abc.abstractmethod
    def _get_loss(self, loss: Union[str, AnyDNN]) -> AnyDNN:
        """Get the loss layer."""
        ...

    @abc.abstractmethod
    def _get_data_loader(
        self,
        data: 'DocumentSequence',
        batch_size: int,
        shuffle: bool,
        num_items_per_class: Optional[int] = None,
        preprocess_fn: Optional['PreprocFnType'] = None,
        collate_fn: Optional['CollateFnType'] = None,
        num_workers: int = 0,
    ) -> AnyDataLoader:
        """Get framework specific data loader from the input data."""
        ...

    @staticmethod
    def _get_num_batches(docs: 'DocumentSequence', batch_size: int) -> int:
        """Get the number of batches from a document sequence"""
        n = len(docs)
        return n // batch_size + 1 * (n % batch_size)

    @abc.abstractmethod
    def _fit(
        self,
        train_data: 'DocumentSequence',
        eval_data: Optional['DocumentSequence'] = None,
        epochs: int = 10,
        batch_size: int = 256,
        num_items_per_class: Optional[int] = None,
        preprocess_fn: Optional['PreprocFnType'] = None,
        collate_fn: Optional['CollateFnType'] = None,
        num_workers: int = 0,
    ):
        """Fit the model (training and evaluation)"""
        ...

    @abc.abstractmethod
    def _train(self, data: AnyDataLoader):
        """Train the model on given labeled data"""
        ...

    def _eval(
        self,
        query_data: 'DocumentSequence',
        index_data: Optional['DocumentSequence'],
        label: str,
        limit: int,
        distance: str,
        batch_size: int,
        preprocess_fn: Optional['PreprocFnType'] = None,
        collate_fn: Optional['CollateFnType'] = None,
    ):
        """Embed the evaluation data and ompute the evaluation metrics"""

        self.state.num_batches_query = self._get_num_batches(query_data, batch_size)
        self.state.num_batches_index = (
            self._get_num_batches(index_data, batch_size) if index_data else 0
        )
        self.state.batch_index = 0

        self._trigger_callbacks('on_val_begin')
        self._trigger_callbacks('on_val_query_begin')

        for idx, batch in enumerate(
            batch_document_sequence(query_data, size=batch_size)
        ):
            self.state.batch_index = idx
            self._trigger_callbacks('on_val_query_batch_begin')
            embed(
                batch,
                self.embed_model,
                device=self._device_name,
                batch_size=batch_size,
                preprocess_fn=preprocess_fn,
                collate_fn=collate_fn,
            )
            self._trigger_callbacks('on_val_query_batch_end')

        self._trigger_callbacks('on_val_query_end')

        if index_data:
            self.state.batch_index = 0
            self._trigger_callbacks('on_val_index_begin')

            for idx, batch in enumerate(
                batch_document_sequence(index_data, size=batch_size)
            ):
                self.state.batch_index = idx
                self._trigger_callbacks('on_val_index_batch_begin')
                embed(
                    batch,
                    self.embed_model,
                    device=self._device_name,
                    batch_size=batch_size,
                    preprocess_fn=preprocess_fn,
                    collate_fn=collate_fn,
                )
                self._trigger_callbacks('on_val_index_batch_end')

            self._trigger_callbacks('on_val_index_end')

        else:
            index_data = query_data

        self._trigger_callbacks('on_val_match_begin')

        evaluator = Evaluator(query_data, index_data)
        self.state.eval_metrics = evaluator.evaluate(
            limit=limit, distance=distance, label=label
        )
        self._trigger_callbacks('on_val_match_end')

        self._trigger_callbacks('on_val_end')
