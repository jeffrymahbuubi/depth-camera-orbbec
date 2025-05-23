from typing import Optional
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class LSTMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LSTMModel`]. It is used to instantiate an
    LSTM model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of a basic LSTM architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        input_size (`int`, *optional*, defaults to 34):
            The dimension of the input features. For pose estimation, this would be the number of keypoints * 2 (x,y coordinates).
        hidden_size (`int`, *optional*, defaults to 128):
            The dimension of the hidden states.
        num_layers (`int`, *optional*, defaults to 1):
            Number of recurrent layers.
        num_labels (`int`, *optional*, defaults to 6):
            The number of labels for classification.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the model.
        bidirectional (`bool`, *optional*, defaults to `False`):
            If `True`, becomes a bidirectional LSTM.
        batch_first (`bool`, *optional*, defaults to `True`):
            If `True`, then the input and output tensors are provided as (batch, seq, feature).
        proj_size (`int`, *optional*, defaults to 0):
            If > 0, will use LSTM with projections of corresponding size.
        window_size (`int`, *optional*, defaults to 32):
            The size of the sliding window for sequential data.
        learning_rate (`float`, *optional*, defaults to 0.001):
            The learning rate for training.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_layer_norm (`bool`, *optional*, defaults to `False`):
            Whether to use layer normalization after the LSTM.
        use_projection (`bool`, *optional*, defaults to `True`):
            Whether to use a projection layer after LSTM.

    Example:

    ```python
    >>> from transformers import LSTMConfig, LSTMModel

    >>> # Initializing a LSTM configuration
    >>> configuration = LSTMConfig()

    >>> # Initializing a model from the configuration
    >>> model = LSTMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "lstm"

    def __init__(
        self,
        input_size=34,
        hidden_size=128,
        num_layers=1,
        num_labels=5,
        dropout=0.0,
        bidirectional=False,
        batch_first=True,
        proj_size=0,
        window_size=16,
        learning_rate=0.001,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_layer_norm=False,
        use_projection=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.proj_size = proj_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_layer_norm = use_layer_norm
        self.use_projection = use_projection