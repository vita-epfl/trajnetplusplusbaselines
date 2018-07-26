import torch
import trajnetbaselines


def test_start_enc():
    input_embedding = trajnetbaselines.lstm.modules.InputEmbedding(2, 4, 1.0)
    obs = torch.Tensor([[0.0, 0.0], [0.0, 0.0]])  # obs with two vel entries
    assert input_embedding.start_enc(obs).numpy().tolist() == [[0, 0, 1, 0], [0, 0, 1, 0]]


def test_start_dec():
    input_embedding = trajnetbaselines.lstm.modules.InputEmbedding(2, 4, 1.0)
    obs = torch.Tensor([[0.0, 0.0], [0.0, 0.0]])  # obs with two vel entries
    assert input_embedding.start_dec(obs).numpy().tolist() == [[0, 0, 0, 1], [0, 0, 0, 1]]
