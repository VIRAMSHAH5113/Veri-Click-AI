from __future__ import annotations

import importlib.util

import pytest


torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(torch_spec is None, reason="PyTorch is an optional deep-learning dependency")


def test_vocabulary_and_title_encoding_are_deterministic():
    from vericlick.deep_model import build_vocab, encode_title, tokenize

    vocab = build_vocab(["Clickbait secret!", "Helpful tutorial"])
    encoded = encode_title("clickbait missing-token!", vocab, max_len=8)
    assert encoded[0] == vocab["clickbait"]
    assert len(encoded) == 8
    assert len(tokenize("hello!")) == 2


def test_fusion_network_forward_output_is_finite():
    import torch
    from vericlick.deep_model import MultimodalFusionNet

    torch.manual_seed(7)
    model = MultimodalFusionNet(vocab_size=32)
    tokens = torch.tensor([[2, 3, 4, 0, 0], [5, 6, 0, 0, 0]], dtype=torch.long)
    images = torch.zeros((2, 3, 96, 96), dtype=torch.float32)
    features = torch.zeros((2, 16), dtype=torch.float32)
    outputs = model(tokens, images, features)
    assert outputs.shape == (2,)
    assert torch.isfinite(outputs).all()


def test_image_tensor_has_expected_shape(sample_png_bytes):
    import torch
    from vericlick.deep_model import image_to_tensor

    image = image_to_tensor(sample_png_bytes)
    assert image.shape == (3, 96, 96)
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
