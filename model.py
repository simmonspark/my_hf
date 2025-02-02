import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Config
import transformers
import torch.nn as nn
import math
import torch.nn.functional as F


class GPT2(nn.Module):
    def __init__(self, config):
        super(GPT2, self).__init__()
        self.embed_dim = config.hidden_size

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, self.embed_dim),
            wpe=nn.Embedding(config.max_position_embedding, self.embed_dim),
            drop=nn.Dropout(config.embd_pdrop),
            h=nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None, pad_mask=None):
        input_embeds = self.transformer.wte(input_ids)
        position_ids = self.transformer.wpe(
            torch.arange(0, input_ids.size(1), device=input_ids.device)).unsqueeze(0)
        hidden_states = input_embeds + position_ids
        hidden_states = self.transformer.drop(hidden_states)
        for block in self.transformer.h:
            hidden_states = block(hidden_states,pad_mask=pad_mask)
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return lm_logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, pad_mask=None, temperature=0.8, do_sample=False, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


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
        attn_out = self.attn(hidden_states, pad_mask)
        hidden_states = attn_out + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim,intermediate_size)
        self.c_proj = nn.Linear(intermediate_size,embed_dim)
        self.act = NewGELU()
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
        max_positions = config.n_positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        self.c_attn = nn.Linear(self.embed_dim,3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states, pad_mask=None):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = query.view(-1, hidden_states.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(-1, hidden_states.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(-1, hidden_states.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        length = key.size(2)

        attn_weights = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        causal_mask = self.bias[:, :, : length, :length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        #attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        attn_weights = attn_weights.masked_fill(causal_mask==False, mask_value)
        if pad_mask is not None:
            if pad_mask.dim() == 2:
                pad_mask = pad_mask.unsqueeze(1).unsqueeze(-1)
            pad_mask = pad_mask.to(dtype=self.dtype)
            pad_mask = (1.0 - pad_mask) * torch.finfo(self.dtype).min
            attn_weights = attn_weights + pad_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(-1, length, self.embed_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output




class MyGPT2Config:
    def __init__(self, model_name="gpt2"):
        tmp_config = GPT2Config.from_pretrained(model_name)
        config_dict = tmp_config.to_dict()
        for key, value in config_dict.items():
            setattr(self, key, value)
        self.hidden_size = 768
        self.max_position_embedding = 1024
        self.embedding_dim = 768
        self.n_layer = 12
        self.attn_pdrop = 0.1

def get_pretrained_model():
    tmp = MyGPT2Config()
    hf = GPT2LMHeadModel.from_pretrained('gpt2')
    print(hf)
    model = GPT2(tmp)
    state = model.state_dict()
    hf_state = hf.state_dict()
    print(state.keys())
    print(hf_state.keys())
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'] # conv 1d 말고 linear로
    for k in state.keys():
        print(k, state[k].shape, hf_state[k].shape)
        if any(k.endswith(w) for w in transposed):
            with torch.no_grad():
                state[k].copy_(hf_state[k].t())
        else:
            with torch.no_grad():
                state[k].copy_(hf_state[k])


    model.load_state_dict(state)
    return model

if __name__ == "__main__":
    model = get_pretrained_model()
    model = model.cuda()
    input_text = "Hello!!!!!!!!!!!!!!!!, my name is"
    prompt = "Hello!!!!!!!!!!!!!!!!, my name is"
    model.to('cuda')
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    encoded_input = tokenizer(prompt, return_tensors='pt').to('cuda')

    x1 = encoded_input['input_ids']

    logits1, loss = model(x1)

    # now draw the argmax samples from each
    y1 = model.generate(x1, max_new_tokens=100, do_sample=False)[0]

    out1 = tokenizer.decode(y1.cpu().squeeze())

    print(out1)