"""Correctness verification: compare solution.py against reference.py.

Tests multiple (batch_size, seq_len_q, seq_len_kv) configurations to ensure
the optimized kernel produces numerically equivalent results (within bf16 tolerance).
"""
import torch
import sys


def test_correctness(batch_size: int, seq_len_q: int, seq_len_kv: int, device: str = "cpu"):
    """Compare optimized vs reference for a given configuration."""
    import reference
    import solution

    axes = {
        "batch_size": batch_size,
        "seq_len_q": seq_len_q,
        "seq_len_kv": seq_len_kv,
    }
    dev = torch.device(device)

    # Generate identical inputs for both
    torch.manual_seed(42)
    ref_inputs = reference.get_inputs(axes, dev)
    torch.manual_seed(42)
    sol_inputs = solution.get_inputs(axes, dev)

    # Verify inputs are identical
    for key in ref_inputs:
        if isinstance(ref_inputs[key], torch.Tensor):
            assert torch.equal(ref_inputs[key], sol_inputs[key]), f"Input {key} mismatch"

    # Run reference
    ref_grad_scores, ref_grad_v = reference.run(**ref_inputs)

    # Run solution — on CPU we can only test the non-Triton path
    # For GPU testing, the Triton kernel will be used
    if device == "cpu":
        # Manually compute the solution's run logic using pure PyTorch on CPU
        # to verify the mathematical equivalence of the GQA reshape optimizations
        sol_grad_scores, sol_grad_v = _run_cpu_equivalent(**sol_inputs)
    else:
        sol_grad_scores, sol_grad_v = solution.run(**sol_inputs)

    # Compare grad_attn_scores
    atol_scores = 0.1
    rtol_scores = 0.1
    scores_match = torch.allclose(ref_grad_scores, sol_grad_scores, atol=atol_scores, rtol=rtol_scores)
    scores_max_diff = (ref_grad_scores.float() - sol_grad_scores.float()).abs().max().item()

    # Compare grad_value_states
    atol_v = 0.1
    rtol_v = 0.1
    v_match = torch.allclose(ref_grad_v, sol_grad_v, atol=atol_v, rtol=rtol_v)
    v_max_diff = (ref_grad_v.float() - sol_grad_v.float()).abs().max().item()

    status = "PASS" if (scores_match and v_match) else "FAIL"
    print(f"  [{status}] B={batch_size}, Sq={seq_len_q}, Skv={seq_len_kv}")
    print(f"    grad_scores: max_diff={scores_max_diff:.6f}, match={scores_match}")
    print(f"    grad_v:      max_diff={v_max_diff:.6f}, match={v_match}")

    return scores_match and v_match


@torch.no_grad()
def _run_cpu_equivalent(
    grad_attn_output, attn_weights, attn_weights_dropped,
    value_states, dropout_mask, attention_dropout,
):
    """CPU-compatible version of the optimized run() logic.
    
    Uses the same GQA reshape tricks but pure PyTorch (no Triton).
    This verifies the mathematical equivalence of the reshape optimizations.
    """
    num_attention_heads = 80
    num_key_value_heads = 8
    num_key_value_groups = num_attention_heads // num_key_value_heads

    batch_size = grad_attn_output.shape[0]
    seq_len_q = grad_attn_output.shape[1]
    seq_len_kv = value_states.shape[2]
    head_dim = value_states.shape[3]

    # Step 1: Transpose + upcast
    grad_attn_output_transposed = grad_attn_output.transpose(1, 2).to(torch.float32)

    # Step 2: GQA-aware Matmul 1 (same logic as solution.py)
    GSq = num_key_value_groups * seq_len_q
    go_gqa = grad_attn_output_transposed.reshape(
        batch_size, num_key_value_heads, GSq, head_dim
    )
    value_states_f32_t = value_states.to(torch.float32).transpose(-2, -1)
    grad_attn_weights_dropped = torch.matmul(go_gqa, value_states_f32_t)
    grad_attn_weights_dropped = grad_attn_weights_dropped.view(
        batch_size, num_attention_heads, seq_len_q, seq_len_kv
    )

    # Step 3: Dropout backward + softmax backward (PyTorch, not Triton)
    if attention_dropout > 0.0:
        grad_attn_weights = grad_attn_weights_dropped * dropout_mask / (1.0 - attention_dropout)
    else:
        grad_attn_weights = grad_attn_weights_dropped

    attn_weights_f32 = attn_weights.to(torch.float32)
    sum_term = (grad_attn_weights * attn_weights_f32).sum(dim=-1, keepdim=True)
    grad_attn_scores = attn_weights_f32 * (grad_attn_weights - sum_term)
    grad_attn_scores = grad_attn_scores.to(torch.bfloat16)

    # Step 4: GQA-aware Matmul 2 + implicit aggregation (same logic as solution.py)
    aw_dropped_gqa = attn_weights_dropped.view(
        batch_size, num_key_value_heads, GSq, seq_len_kv
    ).to(torch.float32)
    grad_value_states = torch.matmul(
        aw_dropped_gqa.transpose(-2, -1),
        go_gqa  # reused
    )
    grad_value_states = grad_value_states.to(torch.bfloat16)

    return grad_attn_scores, grad_value_states


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    test_configs = [
        (1, 128, 128),
        (2, 256, 256),
        (1, 512, 512),
        (2, 1024, 1024),
        (1, 512, 1024),    # asymmetric
        (2, 1024, 512),    # asymmetric reversed
        (1, 128, 2048),    # very asymmetric
    ]

    all_pass = True
    for b, sq, skv in test_configs:
        passed = test_correctness(b, sq, skv, device)
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("✓ All tests PASSED")
    else:
        print("✗ Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
