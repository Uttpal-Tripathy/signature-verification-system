import torch

from sigverify.models.dynamic_branch import DynamicStrokeEncoder
from sigverify.models.fusion import CrossAttentionGatedFusion
from sigverify.models.losses import CombinedEmbeddingLoss
from sigverify.models.static_branch import SiameseCNN


def test_static_branch_embedding_is_unit_norm():
    model = SiameseCNN(backbone="mobilenet_v3_large", embedding_dim=64, pretrained=False)
    model.eval()
    x = torch.rand(2, 1, 224, 224)
    with torch.no_grad():
        embedding = model.embed(x)
    assert embedding.shape == (2, 64)
    norms = embedding.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_static_branch_similarity_range():
    model = SiameseCNN(backbone="mobilenet_v3_large", embedding_dim=32, pretrained=False)
    model.eval()
    with torch.no_grad():
        a = model.embed(torch.rand(3, 1, 224, 224))
        b = model.embed(torch.rand(3, 1, 224, 224))
    sim = model.similarity(a, b)
    assert sim.shape == (3,)
    assert (sim >= -1.0001).all() and (sim <= 1.0001).all()


def test_dynamic_branch_transformer_forward():
    model = DynamicStrokeEncoder(input_dim=7, hidden_dim=32, num_layers=2, num_heads=4, embedding_dim=32, encoder="transformer")
    model.eval()
    seq = torch.rand(4, 50, 7)
    with torch.no_grad():
        embedding, attn = model(seq)
    assert embedding.shape == (4, 32)
    assert attn.shape == (4, 50)
    assert torch.allclose(attn.sum(dim=1), torch.ones(4), atol=1e-4)


def test_dynamic_branch_lstm_forward():
    model = DynamicStrokeEncoder(input_dim=7, hidden_dim=16, num_layers=1, num_heads=2, embedding_dim=24, encoder="lstm", bidirectional=True)
    model.eval()
    seq = torch.rand(2, 30, 7)
    with torch.no_grad():
        embedding, attn = model(seq)
    assert embedding.shape == (2, 24)
    assert attn.shape == (2, 30)


def test_dynamic_branch_hybrid_forward():
    model = DynamicStrokeEncoder(input_dim=7, hidden_dim=16, num_layers=2, num_heads=2, embedding_dim=24, encoder="hybrid", bidirectional=True)
    model.eval()
    seq = torch.rand(2, 30, 7)
    with torch.no_grad():
        embedding, attn = model(seq)
    assert embedding.shape == (2, 24)
    assert attn.shape == (2, 30)
    assert torch.allclose(attn.sum(dim=1), torch.ones(2), atol=1e-4)


def test_static_branch_hybrid_head_embedding_is_unit_norm():
    model = SiameseCNN(backbone="mobilenet_v3_large", embedding_dim=32, pretrained=False, head_type="hybrid", head_kwargs={"num_heads": 4, "num_layers": 1, "max_tokens": 64})
    model.eval()
    x = torch.rand(2, 1, 128, 128)
    with torch.no_grad():
        embedding = model.embed(x)
    assert embedding.shape == (2, 32)
    norms = embedding.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_static_branch_unknown_head_type_raises():
    try:
        SiameseCNN(backbone="mobilenet_v3_large", embedding_dim=16, pretrained=False, head_type="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_fusion_with_and_without_dynamic_modality():
    fusion = CrossAttentionGatedFusion(embedding_dim=16, num_heads=2)
    fusion.eval()
    static_emb = torch.nn.functional.normalize(torch.rand(3, 16), dim=1)
    dynamic_emb = torch.nn.functional.normalize(torch.rand(3, 16), dim=1)

    with torch.no_grad():
        with_dynamic = fusion(static_emb, dynamic_emb)
        without_dynamic = fusion(static_emb, None)

    assert with_dynamic["fused_embedding"].shape == (3, 16)
    assert torch.allclose(without_dynamic["dynamic_weight"], torch.zeros(3), atol=1e-5)
    assert torch.allclose(without_dynamic["static_weight"], torch.ones(3), atol=1e-4)


def test_combined_embedding_loss_is_finite_and_nonnegative():
    criterion = CombinedEmbeddingLoss()
    a = torch.nn.functional.normalize(torch.rand(8, 16), dim=1)
    p = torch.nn.functional.normalize(torch.rand(8, 16), dim=1)
    n = torch.nn.functional.normalize(torch.rand(8, 16), dim=1)
    loss = criterion(a, p, n)
    assert torch.isfinite(loss)
    assert loss.item() >= 0
