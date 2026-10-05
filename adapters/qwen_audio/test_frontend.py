"""Run in the isolated listener venv; validates audio features without model weights."""
from pathlib import Path
from types import SimpleNamespace
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import WhisperFeatureExtractor
from mlx_audio.stt.models.qwen2_audio.qwen2_audio import Qwen2AudioEncoder
from mlx_audio.stt.models.qwen2_audio.config import EncoderConfig

from frontend import install_frontend


def main(model_path):
    mx.random.seed(0)
    encoder = Qwen2AudioEncoder(EncoderConfig(d_model=8, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=16))
    model = SimpleNamespace(audio_tower=encoder, multi_modal_projector=nn.Linear(8, 16),
                            language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=nn.Embedding(8, 16))))
    install_frontend(model, model_path)
    extractor = WhisperFeatureExtractor.from_pretrained(str(model_path), local_files_only=True)
    samples = (.1*np.sin(2*np.pi*440*np.arange(16000*2+11)/16000)).astype(np.float32)
    inputs, count = model._extract_features(mx.array(samples))
    reference = extractor(samples, sampling_rate=16000, return_attention_mask=True, return_tensors="np")
    assert inputs.shape == (1, 128, 3000)
    assert np.max(np.abs(np.array(inputs)-reference['input_features'])) < 1e-6
    valid = (int(reference['attention_mask'].sum())-1)//2+1
    assert valid == 101 and count == 50
    cropped = model.get_audio_features(inputs)
    # Independent full-length attention with padded keys masked out. Its retained
    # positions must equal the adapter's more efficient cropped computation.
    x = nn.gelu(encoder.conv1(inputs.transpose(0,2,1)))
    x = nn.gelu(encoder.conv2(x))
    x = x + encoder.embed_positions[:x.shape[1]]
    mask = (mx.arange(x.shape[1]) < valid)[None,None,None,:]
    for layer in encoder.layers:
        residual = x
        h = layer.self_attn_layer_norm(x)
        attn = layer.self_attn
        b,t,_ = h.shape
        def heads(projection):
            return projection(h).reshape(b,t,attn.num_heads,attn.head_dim).transpose(0,2,1,3)
        h = mx.fast.scaled_dot_product_attention(heads(attn.q_proj), heads(attn.k_proj), heads(attn.v_proj), scale=attn.scale, mask=mask)
        x = residual + attn.out_proj(h.transpose(0,2,1,3).reshape(b,t,attn.embed_dim))
        x = x + layer.fc2(nn.gelu(layer.fc1(layer.final_layer_norm(x))))
    x = x[:, :count*2].reshape(1,count,2,8).mean(axis=2)
    expected = model.multi_modal_projector(encoder.layer_norm(x))
    error = float(mx.max(mx.abs(cropped-expected)))
    assert cropped.shape == (1,50,16) and error < 1e-5, error
    print(f"Reference mel features match; padded-attention equivalence max error={error:.3g}; 50 valid tokens instead of 750.")


if __name__ == '__main__':
    main(Path(sys.argv[1]))
