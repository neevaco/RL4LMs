from typing import Dict, Optional, Tuple

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

# This code is adapted from the implementation for GPT-J provided in the trlX library.
# Original file: https://github.com/CarperAI/trlx/blob/d5a063451a922c43ef0eed8d77beb6d7e1c50c74/examples/summarize_rlhf/reward_model/reward_model.py
#
# TODO(ashwin): if need be, eventually make a PR to include T5 support in the TRLX reward model training scripts.
# Whether this is worthwhile or not is an open question since the way we give inputs to T5 is opinionated and
# intended to be done with a model already fine-tuned for EOA.


def _shift_right(
    bos_token_id: int,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Prepend beginning-of-sequence token (just pad) to provided decoder input ids
    since the decoder expects this. Unlike in conditional generation,
    still keep the end-of-sequence token since we use its hidden state to compute
    the scalar reward.
    """
    old_shape = decoder_input_ids.shape
    shifted_ids = decoder_input_ids.new_zeros((old_shape[0], old_shape[1] + 1))
    shifted_ids[:, 0] = bos_token_id
    shifted_ids[:, 1:] = decoder_input_ids.clone()
    if decoder_attention_mask is None:
        return shifted_ids, None
    old_shape = decoder_attention_mask.shape
    shifted_attention_mask = decoder_attention_mask.new_zeros(
        (old_shape[0], old_shape[1] + 1)
    )
    shifted_attention_mask[:, 0] = 1
    shifted_attention_mask[:, 1:] = decoder_attention_mask.clone()
    return shifted_ids, shifted_attention_mask


def _get_last_hidden_state(
    eos_token_id: int,
    decoder_input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """
    Pull the hidden state associated with the end-of-sequence token
    in each sequence in the decoder_input_ids batch. Before doing this,
    make sure that each sequence has only 1 EOS token.

    Example:
        EOS token = 1
        decoder_input_ids: [[0, 2, 3, 4], [0, 4, 1, 0]]
        output:
            fail, since the first sequence has no EOS token

    Example:
        EOS token = 1
        decoder_input_ids: [[0, 2, 1, 1], [0, 4, 1, 0]]
        output:
            fail, since the first sequenece has 2 EOS tokens

    Example:
        EOS token = 1
        decoder_input_ids: [[0, 2, 3, 1*], [0, 4, 1*, 0]]
        hidden_states:
        [
            [[0.5, 0.5, 0.5], [0.7, 0.7, 0.7], [0.9, 0.9, 0.9], [1.5, 1.5, 1.5]],
            [[2.5, 2.5, 2.5], [2.7, 2.7, 2.7], [2.9, 2.9, 2.9], [3.5, 3.5, 3.5]],
        ]
        output:
        [[1.5, 1.5, 1.5], [2.9, 2.9, 2.9]]
    """
    # (num_matches, 2)
    indices = (decoder_input_ids == eos_token_id).nonzero()

    # make sure that there is exactly one EOS token per row
    target = torch.arange(0, len(indices), 1, dtype=torch.int64).to(
        decoder_input_ids.device
    )
    if (indices[:, 0].shape != target.shape) or not (indices[:, 0] == target).all():
        raise ValueError("there isn't exactly 1 EOS token per sequence in batch")

    # (batch_size, 1, hidden_size)
    indices = (
        indices[:, 1:]
        .unsqueeze(-1)
        .expand((len(decoder_input_ids), 1, hidden_states.shape[-1]))
    )

    # (batch_size, hidden_size)
    last_hidden_states = hidden_states.gather(1, indices).squeeze(1)
    return last_hidden_states


class T5RewardModel(nn.Module):
    """
    A class implementing a T5-based reward model.
    """

    def __init__(self, base_model: AutoModel, tokenizer: AutoTokenizer) -> None:
        """
        Initialize the reward model with the provided base model and tokenizer.
        Also add a linear layer to project hidden states down to a scalar.
        """
        super().__init__()
        self.config = base_model.config
        self.model = base_model
        self.tokenizer = tokenizer
        self.v_head = nn.Linear(base_model.config.d_model, 1, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        chosen_decoder_input_ids: Optional[torch.Tensor] = None,
        chosen_decoder_attention_mask: Optional[torch.Tensor] = None,
        rejected_decoder_input_ids: Optional[torch.Tensor] = None,
        rejected_decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Implement the forward pass for the reward model.

        The encoder takes the prompt (query + sources) as input and the
        decoder takes each of the EOA responses as input. A scalar reward
        is output for each provided response. If pairs are provided, i.e.,
        both "chosen" and "rejected" inputs, then loss is also returned.

        The decoder inputs are prepended with the beginning-of-sequence (BOS) token
        and, unlike in the conditional generation setting, the end-of-sequence (EOS)
        token is also included as an input to the decoder. Here is an illustration
        of the difference:

        Input: x1 x2 x3

        For the generation task, the decoder is given "<bos> x1 x2 x3" and trained to
        output "x1 x2 x3 <eos>".

        For reward modeling, the decoder is given the full "<bos> x1 x2 x3 <eos>" and
        the hidden state for <eos> is project to a scalar reward.
        """
        # just a bunch of input validation
        if input_ids is None:
            raise ValueError("input_ids cannot be None")
        if chosen_decoder_input_ids is None and rejected_decoder_input_ids is None:
            raise ValueError(
                "both chosen_decoder_input_ids and rejected_decoder_input_ids cannot be None"
            )
        if chosen_decoder_input_ids is not None:
            if len(input_ids) != len(chosen_decoder_input_ids):
                raise ValueError(
                    "batch sizes of input_ids and chosen_decoder_input_ids must be same, "
                    f"but len(input_ids)={len(input_ids)} != {len(chosen_decoder_input_ids)}=len(chosen_decoder_input_ids)"
                )
        if rejected_decoder_input_ids is not None:
            if len(input_ids) != len(rejected_decoder_input_ids):
                raise ValueError(
                    "batch sizes of input_ids and rejected_decoder_input_ids must be same, "
                    f"but len(input_ids)={len(input_ids)} != {len(rejected_decoder_input_ids)}=len(rejected_decoder_input_ids)"
                )

        # get eos hidden states for chosen and rejected response inputs if provided
        chosen_scores, rejected_scores = None, None

        if chosen_decoder_input_ids is not None:
            # feed the prompt to the encoder and shifted "chosen" response to the decoder
            chosen_decoder_inputs_shifted = _shift_right(
                self.tokenizer.pad_token_id,
                chosen_decoder_input_ids,
                chosen_decoder_attention_mask,
            )
            # (batch_size, sequence_length, hidden_size)
            chosen_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=chosen_decoder_inputs_shifted[0],
                decoder_attention_mask=chosen_decoder_inputs_shifted[1],
            )[0]

            # (batch_size, hidden_size)
            chosen_hs = _get_last_hidden_state(
                self.tokenizer.eos_token_id,
                chosen_decoder_inputs_shifted[0],
                chosen_outputs,
            )

            # (batch_size,)
            chosen_scores = self.v_head(chosen_hs).squeeze(-1)

        if rejected_decoder_input_ids is not None:
            # feed the prompt to the encoder and shifted "rejected" response to the decoder
            # TODO(ashwin): save encoder hidden states from previous forward pass instead
            # of encoding the same prompts twice
            rejected_decoder_inputs_shifted = _shift_right(
                self.tokenizer.pad_token_id,
                rejected_decoder_input_ids,
                rejected_decoder_attention_mask,
            )

            # (batch_size, sequence_length, hidden_size)
            rejected_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=rejected_decoder_inputs_shifted[0],
                decoder_attention_mask=rejected_decoder_inputs_shifted[1],
            )[0]

            # (batch_size, hidden_size)
            rejected_hs = _get_last_hidden_state(
                self.tokenizer.eos_token_id,
                rejected_decoder_inputs_shifted[0],
                rejected_outputs,
            )

            # (batch_size,)
            rejected_scores = self.v_head(rejected_hs).squeeze(-1)

        outputs = {}

        if chosen_scores is not None:
            outputs["chosen_scores"] = chosen_scores
        if rejected_scores is not None:
            outputs["rejected_scores"] = rejected_scores
        # compute loss if both chosen and rejected responses were provided
        if chosen_scores is not None and rejected_scores is not None:
            outputs["loss"] = -torch.log(
                torch.sigmoid(chosen_scores - rejected_scores)
            ).mean()

        return outputs

def load_t5_reward_model(fname, base_name):
    base_model = AutoModel.from_pretrained(base_name)
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    base_model.resize_token_embeddings(len(tokenizer))
    model = T5RewardModel(base_model, tokenizer)
    model.load_state_dict(torch.load(fname))
    return model