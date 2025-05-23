from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import (
    BaseModelOutput,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

from configuration_lstm import LSTMConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LSTMConfig"

LSTM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # Add pretrained model identifiers here
]


class LSTMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LSTMConfig
    base_model_prefix = "lstm"
    supports_gradient_checkpointing = False
    _no_split_modules = ["LSTMLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class LSTMLayer(nn.Module):
    """LSTM layer with optional layer normalization and dropout."""

    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_size if hasattr(config, 'input_size') else config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=config.batch_first,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            proj_size=config.proj_size if config.proj_size > 0 else 0,
        )
        
        self.use_layer_norm = config.use_layer_norm
        if self.use_layer_norm:
            norm_size = config.hidden_size * (2 if config.bidirectional else 1)
            self.layer_norm = nn.LayerNorm(norm_size, eps=config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_tensor,
        hidden_states=None,
        cell_states=None,
    ):
        lstm_output, (hidden_states, cell_states) = self.lstm(
            input_tensor, 
            (hidden_states, cell_states) if hidden_states is not None else None
        )
        
        if self.use_layer_norm:
            lstm_output = self.layer_norm(lstm_output)
        
        lstm_output = self.dropout(lstm_output)
        
        return lstm_output, (hidden_states, cell_states)


LSTM_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LSTMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LSTM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.FloatTensor` of shape `(batch_size, sequence_length, input_size)`):
            Input sequence tensor.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LSTM Model transformer outputting raw hidden-states without any specific head on top.",
    LSTM_START_DOCSTRING,
)
class LSTMModel(LSTMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.lstm_layer = LSTMLayer(config)
        
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(LSTM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import LSTMConfig, LSTMModel
        >>> import torch

        >>> # Initializing a LSTM configuration
        >>> configuration = LSTMConfig()

        >>> # Initializing a model from the configuration
        >>> model = LSTMModel(configuration)

        >>> # Random input tensor (batch_size=2, sequence_length=32, input_size=34)
        >>> inputs = torch.randn(2, 32, 34)
        >>> outputs = model(inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention mask
            attention_mask = attention_mask.unsqueeze(-1).expand_as(input_ids)
            input_ids = input_ids * attention_mask

        # Pass through LSTM
        sequence_output, (final_hidden, final_cell) = self.lstm_layer(input_ids)

        # Get the last hidden state
        if self.config.bidirectional:
            # Concatenate forward and backward hidden states
            last_hidden_state = torch.cat([final_hidden[-2], final_hidden[-1]], dim=-1)
        else:
            last_hidden_state = final_hidden[-1]

        if not return_dict:
            return (sequence_output, last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=(last_hidden_state,) if output_hidden_states else None,
        )


@add_start_docstrings(
    """LSTM Model with a classification head on top (a linear layer on top of the hidden-states output).""",
    LSTM_START_DOCSTRING,
)
class LSTMForSequenceClassification(LSTMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.lstm = LSTMModel(config)
        
        # Classification head
        classifier_input_size = config.hidden_size * (2 if config.bidirectional else 1)
        
        if config.use_projection:
            self.pre_classifier = nn.Linear(classifier_input_size, config.hidden_size)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.dropout = nn.Dropout(config.dropout)
        else:
            self.classifier = nn.Linear(classifier_input_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(LSTM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Example:

        ```python
        >>> from transformers import LSTMConfig, LSTMForSequenceClassification
        >>> import torch

        >>> # Number of labels for classification
        >>> num_labels = 6

        >>> # Initializing a LSTM configuration
        >>> configuration = LSTMConfig(num_labels=num_labels)

        >>> # Initializing a model from the configuration
        >>> model = LSTMForSequenceClassification(configuration)

        >>> # Random input tensor (batch_size=2, sequence_length=32, input_size=34)
        >>> inputs = torch.randn(2, 32, 34)
        >>> labels = torch.tensor([1, 3])

        >>> outputs = model(inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.lstm(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get the last hidden state from LSTM
        if return_dict:
            # For sequence classification, we typically use the last hidden state
            sequence_output = outputs.last_hidden_state
            if len(outputs.hidden_states) > 0:
                pooled_output = outputs.hidden_states[0]  # Final hidden state
            else:
                # If hidden states not returned, use the last timestep
                pooled_output = sequence_output[:, -1, :]
        else:
            sequence_output = outputs[0]
            pooled_output = outputs[1]

        # Apply projection if configured
        if self.config.use_projection:
            pooled_output = self.pre_classifier(pooled_output)
            pooled_output = torch.relu(pooled_output)
            pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=None,  # LSTM doesn't have attention weights
        )


# Add to AutoModel registry
def register_lstm_auto_model():
    """Register the LSTM model with AutoModel."""
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
        
        AutoConfig.register("lstm", LSTMConfig)
        AutoModel.register(LSTMConfig, LSTMModel)
        AutoModelForSequenceClassification.register(LSTMConfig, LSTMForSequenceClassification)
    except ImportError:
        logger.warning("Could not register LSTM model with AutoModel. Please ensure transformers is installed.")


# Register on module import
register_lstm_auto_model()

__all__ = [
    "LSTMConfig",
    "LSTMPreTrainedModel", 
    "LSTMModel",
    "LSTMForSequenceClassification",
]