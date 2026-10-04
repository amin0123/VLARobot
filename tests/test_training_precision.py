import torch

from src.training.precision import promote_trainable_parameters, restore_checkpoint_precision


def test_small_optimizer_updates_survive_promotion_without_changing_frozen_weights():
    module = torch.nn.Linear(2, 2, bias=True).to(torch.bfloat16)
    with torch.no_grad():
        module.weight.fill_(1)
        module.bias.fill_(2)
    module.bias.requires_grad_(False)
    original_bias = module.bias.detach().clone()
    parameter = module.weight
    assert promote_trainable_parameters(module) == 4
    assert module.weight is parameter
    assert module.weight.dtype == torch.float32
    assert module.bias.dtype == torch.bfloat16
    assert torch.equal(module.bias, original_bias)

    optimizer = torch.optim.AdamW([module.weight], lr=1e-5, weight_decay=0)
    module.weight.sum().backward()
    optimizer.step()
    # At magnitude one, this update would round back to one in BF16 storage.
    assert torch.all(module.weight < 1)
    assert torch.all(module.weight.to(torch.bfloat16) == 1)
    assert optimizer.state[module.weight]["exp_avg"].dtype == torch.float32
    assert optimizer.state[module.weight]["exp_avg_sq"].dtype == torch.float32
    assert torch.equal(module.bias, original_bias)


def test_checkpoint_load_preserves_fp32_values_despite_bf16_constructor(tmp_path):
    from safetensors.torch import load_model, save_model

    source = torch.nn.Linear(2, 2)
    with torch.no_grad():
        source.weight.fill_(0.99999)
    save_model(source, tmp_path / "model.safetensors")
    target = torch.nn.Linear(2, 2).to(torch.bfloat16)
    load_model(target, tmp_path / "model.safetensors")
    assert torch.all(target.weight == 1)  # Default load rounded away the small update.
    parameter = target.weight
    assert restore_checkpoint_precision(target, tmp_path) == 6
    assert target.weight is parameter
    assert target.weight.dtype == torch.float32
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)
    assert restore_checkpoint_precision(target, tmp_path) == 0
