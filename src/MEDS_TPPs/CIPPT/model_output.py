"""Classes and utilities for model output layers.

Attributes:
    BERNOULLI_DIST_T: The type of a bernoulli distribution.
    REGRESSION_DIST_T: The type of a regression distribution.
"""
from dataclasses import dataclass

import torch
from transformers.utils import ModelOutput

from .config import StructuredTransformerConfig, TimeToEventGenerationHeadType
from .generative_layers import (
    ExponentialTTELayer,
    GaussianIndexedRegressionLayer,
    LogNormalMixtureTTELayer,
)
from .utils import safe_weighted_avg, str_summary, weighted_loss

BERNOULLI_DIST_T = torch.distributions.Bernoulli
REGRESSION_DIST_T = torch.distributions.Normal


@dataclass
class TransformerOutputWithPast(ModelOutput):
    """Holds output data from a transformer model.

    This class is designed to manage output data from a transformer model,
    which may include last hidden state, past key values, hidden states, and attentions.

    Args:
        last_hidden_state: The last hidden state from the model.
        past_key_values: The past key values from the model.
        hidden_states: The hidden states from the model.
        attentions: The attentions from the model.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: tuple[tuple[torch.FloatTensor]] | dict[str, tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@dataclass
class GenerativeSequenceModelLosses(ModelOutput):
    """Holds losses data for a Generative Sequence Model.

    This class is designed to manage losses from a Generative Sequence Model,
    which can include classification, regression and time to event losses.

    Args:
        classification: Losses for the classification task.
        regression: Losses for the regression task.
        time_to_event: Loss for the time-to-event task.
    """

    classification: torch.FloatTensor | None = None
    regression: torch.FloatTensor | None = None
    time_to_event: torch.FloatTensor | None = None


@dataclass
class GenerativeSequenceModelPredictions(ModelOutput):
    """Contains the predictions for the GenerativeSequenceModel head.

    Args:
        classification: The predicted classification task results.
        regression: The predicted regression task results.
        regression_indices: The predicted indices for the regression task.
        time_to_event: The predicted time-to-event results.
    """

    classification: BERNOULLI_DIST_T | None = None
    regression: REGRESSION_DIST_T | None = None
    regression_indices: torch.LongTensor | None = None
    time_to_event: torch.distributions.Distribution | None = None


@dataclass
class GenerativeSequenceModelLabels(ModelOutput):
    """Contains the labels for the GenerativeSequenceModel head.

    The labels are split by task type. Single-label classification task labels will have
    shape batch X seq and have raw integer labels, whereas multi-label classification task labels
    will have shape batch X seq X vocab size and have binary indicators for each label.

    Args:
        classification: The classification task labels.
        regression: The regression task labels.
        regression_indices: The indices for the regression task.
        time_to_event: The time-to-event task labels.
    """

    classification: torch.LongTensor | None = None
    regression: torch.FloatTensor | None = None
    regression_indices: torch.LongTensor | None = None
    time_to_event: torch.FloatTensor | None = None


@dataclass
class GenerativeSequenceModelOutput(ModelOutput):
    """Contains all GenerativeSequenceModel outputs.

    The outputs include losses, predictions, labels, and masks, among others.

    Args:
        loss: The overall model loss.
        losses: The specific model losses by task type.
        preds: The model predictions.
        labels: The model labels.
        event_mask: A boolean tensor representing the event mask.
        dynamic_values_mask: A boolean tensor representing the dynamic values mask.
        past_key_values: The past key values from the model.
        hidden_states: The hidden states from the model.
        attentions: The attentions from the model.
    """

    loss: torch.FloatTensor
    losses: GenerativeSequenceModelLosses | None = None
    preds: GenerativeSequenceModelPredictions | None = None
    labels: GenerativeSequenceModelLabels | None = None
    event_mask: torch.BoolTensor | None = None
    dynamic_values_mask: torch.BoolTensor | None = None

    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@dataclass
class StreamClassificationModelOutput(ModelOutput):
    """Contains all outputs for the Stream Classification Model.

    Args:
        loss: The overall model loss.
        preds: The model predictions.
        labels: The model labels.
    """

    loss: torch.FloatTensor
    preds: torch.FloatTensor = None
    labels: torch.LongTensor | torch.FloatTensor = None


class GenerativeOutputLayerBase(torch.nn.Module):
    """A base class for the output layer of a generative model.

    This class is responsible for constructing the time-to-event (TTE) layer based on the
    TTE_generation_layer_type in the given config, along with observation and classification layers. It also
    establishes the criteria for observation and classification. It does not contain a forward method which
    actually calls these helper methods, as those are implemented by subclass specific methods depending on
    how the encoded state is structured.

    This class should not be instantiated directly. Instead, use one of the derived classes.

    Args:
        config: A configuration object of type StructuredTransformerConfig.

    Raises:
        ValueError: If the TTE_generation_layer_type in the config is not valid.
    """

    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__()

        self.config = config

        match self.config.TTE_generation_layer_type:
            case TimeToEventGenerationHeadType.LOG_NORMAL_MIXTURE:
                self.TTE_layer = LogNormalMixtureTTELayer(
                    in_dim=config.hidden_size,
                    num_components=config.TTE_lognormal_generation_num_components,
                    mean_log_inter_time=config.mean_log_inter_event_time_min,
                    std_log_inter_time=config.std_log_inter_event_time_min,
                )
            case TimeToEventGenerationHeadType.EXPONENTIAL:
                self.TTE_layer = ExponentialTTELayer(in_dim=config.hidden_size)
            case _:
                raise ValueError(
                    f"Invalid option for `config.TTE_generation_layer_type`. Must be "
                    f"a member of the `TimeToEventGenerationHeadType` enum: "
                    f"({TimeToEventGenerationHeadType.values()}). got {config.TTE_generation_layer_type}."
                )

        self.ClassificationLayer = torch.nn.Linear(config.hidden_size, config.vocab_size)
        self.classification_criteria = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.RegressionLayer = GaussianIndexedRegressionLayer(
            n_regression_targets=config.vocab_size,
            in_dim=config.hidden_size,
        )

    def get_TTE_outputs(
        self, batch: dict, encoded: torch.FloatTensor, is_generation: bool = False
    ) -> tuple[torch.FloatTensor, torch.distributions.Distribution, torch.FloatTensor,]:
        """Produces time-to-event predictions and log likelihoods (**not NLLs!**) for the model.

        Args:
            batch: The batch of data for which the classification predictions are desired.
            encoded: The final encodings used to predict the time from the event at a position to the
                subsequent event. This tensor is of shape (batch size X sequence length X hidden dim).
            is_generation: A boolean to indicate if the function is used for generation. Defaults to False. If
                true, then the model will only return the predicted distribution (as that is all that is used
                in generative use-cases).

        Returns:
            A tuple containing the following items:
                TTE_LL: A torch scalar containing the average log-likelihood of observed time-to-events given
                the predicted distribution.
                TTE_dist: The predicted torch Distribution for modelling time-to-event.
                TTE_true: A tensor containing the observed time between events for each batch element.

        Raises:
            ValueError: If NaNs are found in TTE_obs_mask_exp, TTE_true_exp or TTE_LL or if there is no
            observed time-to-event for >= 1 patient in the batch.
        """
        TTE_dist = self.TTE_layer(encoded)

        if is_generation:
            return None, TTE_dist, None

        # TTE_dist is a distribution with random variables of shape (batch size, sequence length)
        TTE_obs_mask = batch["event_mask"][:, 1:] & batch["event_mask"][:, :-1]
        TTE_delta = batch["time_delta"][:, :-1]
        TTE_true = torch.where(TTE_obs_mask, TTE_delta, torch.ones_like(TTE_delta))

        # As TTE_dist contains a predicted distribution for the last sequence element, which we want to return
        # for generative purposes, we add a fake observation to the last element.
        TTE_true_exp = torch.cat((TTE_true, torch.ones_like(TTE_true[:, -1]).unsqueeze(-1)), dim=-1)
        TTE_obs_mask_exp = torch.cat(
            (TTE_obs_mask, torch.zeros_like(TTE_obs_mask[:, -1]).unsqueeze(-1)), dim=-1
        )

        # We skip the last event as we have no true time to event for that event.
        # TODO(mmd): Use NLL-\beta?
        try:
            TTE_LL = TTE_dist.log_prob(TTE_true_exp)
        except ValueError as e:
            raise ValueError(f"Failed to compute TTE log prob on input {str_summary(TTE_true_exp)}") from e

        if TTE_obs_mask_exp.isnan().any():
            raise ValueError(f"NaNs in TTE_obs_mask_exp: {batch}")
        elif TTE_true_exp.isnan().any():
            raise ValueError(f"NaNs in TTE_true_exp: {batch}")
        elif TTE_LL.isnan().any():
            raise ValueError(f"NaNs in TTE_LL: {batch}")
        elif (TTE_obs_mask_exp.float().sum(-1) == 0).any():
            raise ValueError(f"No observed time-to-event for >= 1 patient in batch: {batch}")

        TTE_LL_per_patient = (TTE_LL * TTE_obs_mask_exp.float()).sum(-1) / TTE_obs_mask_exp.float().sum(-1)
        TTE_LL_overall = TTE_LL_per_patient.mean()

        return TTE_LL_overall, TTE_dist, TTE_true

    def get_classification_output(
        self,
        batch: dict,
        encoded: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, BERNOULLI_DIST_T, torch.FloatTensor]:
        """Produces classification predictions and losses for the model.

        Args:
            batch: The batch of data for which the classification predictions are desired.
            encoded: The final encodings *to be used to predict for each position in the sequence*. For
                example, the vector ``encoded[i][j]`` (which is of size ``hidden_dim``) is *not* the summary
                encoding of the batch element at batch index ``i`` and sequence index ``j``, but rather is the
                input to be used to form classification predictions corresponding to batch element ``i`` at
                sequence position ``j``.

        Returns:
            The following three tensors: losses, prediction distributions, labels
        """

        torch._assert(~torch.isnan(encoded).any(), f"{torch.isnan(encoded).sum()} NaNs in encoded")

        # Classification of what elements are going to occur:
        scores = self.ClassificationLayer(encoded)

        labels = (
            torch.zeros(scores.shape[0], scores.shape[1], 1 + scores.shape[2], device=scores.device).scatter(
                dim=2, index=batch["dynamic_indices"], value=1
            )
        )[:, :, 1:]

        loss_per_label = self.classification_criteria(scores, labels)
        loss_per_event = loss_per_label.mean(dim=-1)

        dists = torch.distributions.Bernoulli(logits=scores)

        loss_overall = weighted_loss(loss_per_event, batch["event_mask"])

        return loss_overall, dists, labels

    def get_regression_outputs(
        self,
        batch: dict,
        encoded: torch.FloatTensor,
        is_generation: bool = False,
    ) -> tuple[torch.FloatTensor, torch.distributions.Distribution, torch.FloatTensor, torch.LongTensor,]:
        """Produces regression predictions and losses for the model.

        Args:
            batch: The batch of data for which the regression predictions are desired.
            encoded: The final encodings (of shape batch_size X sequence_length X hidden_dim) **to be used to
                predict for each position in the sequence**. For example, the vector `encoded[i][j]` (which is
                of size `hidden_dim`) is _not_ the summary encoding of the batch element at batch index `i`
                and sequence index `j`, but rather is the input to be used to form regression predictions
                corresponding to batch element `i` at sequence position `j`.

        Returns:
            The loss, the distributions, the labels, and the indices over codes of which codes the value
            labels correspond to.
        """

        indices_measured_or_zero = torch.where(
            batch["dynamic_values_mask"],
            batch["dynamic_indices"],
            torch.zeros_like(batch["dynamic_indices"]),
        ).long()

        regr_dist = self.RegressionLayer(X=encoded, idx=(None if is_generation else indices_measured_or_zero))

        if is_generation:
            return None, regr_dist, None, None

        values_observed_or_zero = torch.where(
            batch["dynamic_values_mask"],
            batch["dynamic_values"],
            torch.zeros_like(batch["dynamic_values"]),
        ).float()

        # We don't need to shift here, as given this is a structured model, we'll always rely on elements
        # of the dependency graph that don't include these inputs to predict them (e.g., predict the
        # contents of the event given the time at which the event occurred).

        # TODO(mmd): Use NLL-\beta?
        loss_per_label = -regr_dist.log_prob(values_observed_or_zero)
        loss_per_event, _ = safe_weighted_avg(loss_per_label, batch["dynamic_values"])

        events_with_label = batch["event_mask"] & batch["dynamic_values"].any(dim=-1)
        loss_overall = weighted_loss(loss_per_event, events_with_label)

        return loss_overall, regr_dist, values_observed_or_zero, indices_measured_or_zero
