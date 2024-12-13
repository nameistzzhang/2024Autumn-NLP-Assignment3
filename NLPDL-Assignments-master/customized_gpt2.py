from turtle import forward
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model, GPT2LMHeadModel

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """
    # ? This class is copied from Huggingface's transformers library, we didn't further modify it. We paste it here only to get a better understanding of the model.
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def __repr__(self) -> str:
        return "Conv1D(nf={nf}, nx={nx})".format(**self.__dict__)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

class CustomizedGPT2Attention(GPT2Attention):
    """
    GPT2 flash attention module. This module inherits from `GPT2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_kvcache = None
        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        # * attention_kvcache being delivered from the GPT Block
        self.attn_kvcache = kwargs.get('attn_kvcache', None)

        if self.attn_kvcache['key'] is None:    # * If the cache is empty, then it is the first iteration
        # Prepare query, key, value matrix
            query, key, value = self.c_attn(hidden_states).split(self.embed_dim, dim=2)
        else:    # * If the cache is not empty, then it is not the first iteration, we use kv_cache to update the key and value
            query, new_key, new_value = self.c_attn(hidden_states[:, -1:, :].view(-1, 1, self.embed_dim)).split(self.embed_dim, dim=2)
            key = torch.cat([self.attn_kvcache['key'], new_key], dim=1)
            value = torch.cat([self.attn_kvcache['value'], new_value], dim=1)
            
        #* Update the cache
        self.attn_kvcache['key'] = key
        self.attn_kvcache['value'] = value
            
            
        query = self._split_heads(query, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]
        key = self._split_heads(key, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]
        value = self._split_heads(value, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]

        # Self-attention mechanism
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim) # [batch_size, seq_len, dim]
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, self.attn_kvcache


class CustomizedGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.attn = CustomizedGPT2Attention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        residual = hidden_states

        # self-attention (class `CustomizedGPT2AttentionWithFasterCache`)
        hidden_states = self.ln_1(hidden_states)
        attn_output, attn_kvcache = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )    #* Passing KV_cache to the attention module

        # residual connection
        hidden_states = attn_output + residual


        residual = hidden_states

        # feed-forward
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states


        return hidden_states, attn_kvcache


class CustomizedGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.h = nn.ModuleList([CustomizedGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        assert self._attn_implementation == 'eager', "[NLPDL ERROR] set _attn_implementation to either 'eager' or 'faster_cache' in this version"

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        
        #* KV_cache is a list of dictionaries, each dictionary contains the key and value for the corresponding layer
        KV_cache = kwargs.get('KV_cache', None)
        
        if KV_cache is None:
            KV_cache = [{'key': None, 'value': None} for i in range(self.config.num_hidden_layers)]

        input_shape = input_ids.size()    # * input_ids shape: (batchsize, seqlen)
        batch_size = input_ids.shape[0]    # * attention_mask shape: (batchsize, seqlen)
        device = input_ids.device

        # Prepare input embeddings
        inputs_embeds = self.wte(input_ids)    # * inputs_embeds shape: (batchsize, seqlen, embeddim)
        position_ids = attention_mask.long().cumsum(-1) - 1    # using accumulated sum to get position_ids (-1 for starting from 0)
        position_ids.masked_fill_(attention_mask == 0, 1)    # replace 1 for padding tokens
        position_embeds = self.wpe(position_ids)    # * position_embeds shape: (batchsize, seqlen, embeddim)
        hidden_states = inputs_embeds + position_embeds # * hidden_states shape: (batchsize, seqlen, embeddim)

        # Prepare Attention mask.
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        # Iterate over all GPT2 layer, i.e. `block`
        for i, block in enumerate(self.h):
            outputs, attn_kvcache = block(
                hidden_states,
                attention_mask=attention_mask,
                attn_kvcache = KV_cache[i]    #* Get the appropriate KV_cache for the layer
            )

            hidden_states = outputs

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return hidden_states, KV_cache


class CustomizedGPT2LMHeadModel(GPT2LMHeadModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomizedGPT2Model(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        hidden_states, KV_cache = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # Prepare logits from last hidden state
        lm_logits = self.lm_head(hidden_states)

        return {
            'logits': lm_logits,
            'KV_cache': KV_cache
        }    # * Return KV_cache for the next iteration