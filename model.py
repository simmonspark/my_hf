import torch
from sympy.crypto import padded_key
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import GPT2Config
import transformers
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
import math
import torch.nn.functional as F


class GPT2(nn.Module):
    def __init__(self, config):
        super(GPT2, self).__init__(config)
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embedding, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, labels=None, pad_mask=None):
        input_embeds = self.wte(input_ids)
        position_ids = self.wpe(
            torch.arange(0, input_ids.size(1), input_ids.size(1), device=input_ids.device, dtype=torch.long))
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        hidden_states = input_embeds + position_ids
        hidden_states = self.drop(hidden_states)
        hidden_states = self.h(hidden_states, pad_mask)
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss
        return lm_logits



class GPT2Block(nn.Module):
    def __init__(self, config):
        super(GPT2Block, self).__init__()
        hidden_size = config.hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config=config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config.hidden_size * 4, config)

    def forward(self, hidden_states, pad_mask=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_out = self.attn(hidden_states,pad_mask)
        hidden_states = attn_out + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Conv1d(intermediate_size, embed_dim)
        self.c_proj = nn.Conv1d(embed_dim, intermediate_size)
        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Attention(nn.Module):
    def __init__(self, config):
        super(GPT2Attention, self).__init__()
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        self.c_attn = nn.Conv1d(3 * self.embed_dim, self.embed_dim)
        self.c_proj = nn.Conv1d(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states, pad_mask=None):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = query.view(-1, hidden_states.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(-1, hidden_states.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(-1, hidden_states.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        length = key.size(1)
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / self.head_dim
        causal_mask = self.bias[1, 1, : length, length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        if pad_mask is not None:
            if pad_mask.dim() == 2:
                pad_mask = pad_mask.unsqueeze(1).unsqueeze(-1)
            pad_mask = pad_mask.to(dtype=self.dtype)
            pad_mask = (1.0 - pad_mask) * torch.finfo(self.dtype).min
        attn_weights = attn_weights + pad_mask
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(-1, length, self.embed_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output
