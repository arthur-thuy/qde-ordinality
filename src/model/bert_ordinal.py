"""Module for building ordinal BERT."""

# standard library imports
from typing import Optional, Tuple, Union

# related third party imports
import torch
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss, corn_loss
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from yacs.config import CfgNode

# local application/library specific imports
from model.build import ORDINAL_MODEL_REGISTRY
from tools.constants import OrdinalType, OutputType
from model.ordered_logit import OrderedLogitLayer, cumulative_link_loss

# NOTE: model from `modeling_bert.py`
# https://github.com/huggingface/transformers/blob/4831a94ee7d897c15f2dbf93dd039d36d9fdc61a/src/transformers/models/bert/modeling_bert.py#L1620


@ORDINAL_MODEL_REGISTRY.register("bert-base-uncased")
def build_bert_ordinal(model_cfg: CfgNode, output_type: OutputType) -> PreTrainedModel:
    """Build BERT model for ordinal regression.

    Parameters
    ----------
    model_cfg : CfgNode
        Model config object
    output_type : OutputType
        Output type

    Returns
    -------
    PreTrainedModel
        HF model
    """
    model = BertForSequenceOrdinal.from_pretrained(
        pretrained_model_name_or_path=model_cfg.NAME,
        num_labels=model_cfg.NUM_LABELS,
        problem_type=output_type,
        torch_dtype="auto",
    )
    return model


class BertForSequenceOrdinal(BertPreTrainedModel):
    """BERT model for ordinal regression."""

    def __init__(self, config: PretrainedConfig):
        """Initialize the model.

        Parameters
        ----------
        config : PretrainedConfig
            _description_

        Raises
        ------
        ValueError
            If `output_type` is not specified in the config
        NotImplementedError
            If the output type is not implemented
        """
        super().__init__(config)
        if self.config.problem_type is None:
            raise ValueError("`output_type` must be specified in the config")
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # output layer
        if config.problem_type == OrdinalType.CORAL:
            # NOTE: layer does num_classes-1 nodes
            self.classifier = CoralLayer(config.hidden_size, config.num_labels)
        elif config.problem_type == OrdinalType.CORN:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels - 1)
        elif config.problem_type == OrdinalType.OR_NN:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels - 1)
        elif config.problem_type == OrdinalType.LOGIT:
            self.classifier = OrderedLogitLayer(
                size_in=config.hidden_size,
                num_classes=config.num_labels,
                init_cutpoints="equal_density",  # "equal_spaced",  # TODO: make dynamic!
            )
        else:
            raise NotImplementedError(
                f"Output type {config.problem_type} is not implemented"
            )

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, OrderedLogitLayer):
            # Let OrderedLogitLayer initialize itself
            module.reset_parameters()
        else:
            # Use default BERT initialization for other layers
            super()._init_weights(module)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        try:
            self.config.problem_type = OutputType(self.config.problem_type)
        except ValueError:
            raise ValueError(f"Invalid output type: {self.config.problem_type}")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # NOTE: outputs[1] takes "pooled_output" from BERT
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # print(f"{logits.shape = }")  # TODO: remove
        # print(f"{labels.shape = }")  # TODO: remove

        loss = None
        if labels is not None:
            if self.config.problem_type == OutputType.ORD_CORAL:
                levels = levels_from_labelbatch(labels, num_classes=self.num_labels)
                levels = levels.to(logits.device)
                loss = coral_loss(logits, levels)  # NOTE: no need to flatten
                # print(f"{levels.shape = }")  # TODO: remove
                # print(f"{loss = }")  # TODO: remove
            elif self.config.problem_type == OutputType.ORD_CORN:
                loss = corn_loss(
                    logits, labels, num_classes=self.num_labels
                )  # NOTE: no need to flatten
                # print(f"{loss = }")  # TODO: remove
            elif self.config.problem_type == OutputType.ORD_OR_NN:
                levels = levels_from_labelbatch(labels, num_classes=self.num_labels)
                levels = levels.to(logits.device)
                loss = coral_loss(logits, levels)  # NOTE: same loss as CORAL!
                # print(f"{levels.shape = }")  # TODO: remove
                # print(f"{loss = }")  # TODO: remove
            elif self.config.problem_type == OutputType.ORD_LOGIT:
                # NOTE: for ORD_LOGIT, `logits` are the probabilities for each class
                loss = cumulative_link_loss(logits, labels)
                # print(f"{loss = }")  # TODO: remove

        if not return_dict:
            # NOTE: outputs[2:] takes "encoder_outputs[:1]" from BERT
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
