"""Reference-compatible Qwen audio features for the pinned MLX-Audio revision.

That revision uses an HTK mel bank instead of Whisper's Slaney bank, retains an
extra STFT frame, and treats padded silence as 750 real audio tokens. Use the
installed Transformers feature extractor and discard padded encoder positions.
Cropping AFTER the convolutions preserves their boundary context; self-attention
over the retained positions equals masking padded keys in the full sequence.
"""
from types import MethodType


FRONTEND_VERSION = "whisper-slaney-valid-length-v1"


def install_frontend(model, model_path):
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor.from_pretrained(str(model_path), local_files_only=True)
    state = {}

    def extract(self, audio):
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        batch = extractor(samples, sampling_rate=16000, return_attention_mask=True, return_tensors="np")
        frames = int(batch["attention_mask"].sum())
        valid = (frames - 1) // 2 + 1
        state["valid_encoder_frames"] = valid
        return mx.array(batch["input_features"], dtype=self.audio_tower.conv1.weight.dtype), valid // 2

    def features(self, inputs):
        enc = self.audio_tower
        x = nn.gelu(enc.conv1(inputs.transpose(0, 2, 1)))
        x = nn.gelu(enc.conv2(x))
        valid = state["valid_encoder_frames"]
        x = x[:, :valid] + enc.embed_positions[:valid]
        for layer in enc.layers:
            x = layer(x)
        b, length, width = x.shape
        x = x[:, :(length // 2) * 2].reshape(b, length // 2, 2, width).mean(axis=2)
        x = enc.layer_norm(x)
        return self.multi_modal_projector(x).astype(self.language_model.model.embed_tokens.weight.dtype)

    model._extract_features = MethodType(extract, model)
    model.get_audio_features = MethodType(features, model)
    return model
