import torch


# ---------------------------------------------------------------------------
# Fused dropout backward + softmax backward (torch.compile)
# ---------------------------------------------------------------------------
# CRITICAL: dynamic=True allows one compilation to handle ALL input shapes.
# With dynamic=False, each new shape triggers recompilation and hits the
# recompile_limit (8), causing fallback to slow eager mode for remaining shapes.
# ---------------------------------------------------------------------------

@torch.compile(dynamic=True)
def _fused_dropout_softmax_bwd(
    grad_aw_dropped: torch.Tensor,   # (B, H, Sq, Skv) bf16
    attn_weights: torch.Tensor,       # (B, H, Sq, Skv) bf16
    dropout_mask: torch.Tensor,       # (B, H, Sq, Skv) bool
    inv_1_minus_p: float,
) -> torch.Tensor:
    """Fused dropout_bwd + softmax_bwd. All f32 math in registers."""
    grad_aw = grad_aw_dropped.float() * dropout_mask * inv_1_minus_p
    aw_f32 = attn_weights.float()
    sum_term = (grad_aw * aw_f32).sum(dim=-1, keepdim=True)
    return (aw_f32 * (grad_aw - sum_term)).to(torch.bfloat16)


@torch.compile(dynamic=True)
def _fused_softmax_bwd_no_dropout(
    grad_aw_dropped: torch.Tensor,
    attn_weights: torch.Tensor,
) -> torch.Tensor:
    """Fused softmax_bwd (no dropout variant)."""
    grad_aw = grad_aw_dropped.float()
    aw_f32 = attn_weights.float()
    sum_term = (grad_aw * aw_f32).sum(dim=-1, keepdim=True)
    return (aw_f32 * (grad_aw - sum_term)).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# get_inputs
# ---------------------------------------------------------------------------

def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs. grad_attn_output in (B, H, Sq, D) layout to skip transpose."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len_q = axes_and_scalars["seq_len_q"]
    seq_len_kv = axes_and_scalars["seq_len_kv"]
    num_attention_heads = 80
    num_key_value_heads = 8
    head_dim = 128
    attention_dropout = 0.1
    
    grad_attn_output = torch.randn(
        batch_size, num_attention_heads, seq_len_q, head_dim,
        dtype=torch.bfloat16, device=device
    )
    
    attn_scores_raw = torch.randn(
        batch_size, num_attention_heads, seq_len_q, seq_len_kv,
        dtype=torch.float32, device=device
    )
    attn_weights = torch.softmax(attn_scores_raw, dim=-1).to(torch.bfloat16)
    
    dropout_mask = torch.rand(
        batch_size, num_attention_heads, seq_len_q, seq_len_kv,
        device=device
    ) > attention_dropout
    
    if attention_dropout > 0.0:
        attn_weights_dropped = (attn_weights.float() * dropout_mask / (1.0 - attention_dropout)).to(torch.bfloat16)
    else:
        attn_weights_dropped = attn_weights
    
    value_states = torch.randn(
        batch_size, num_key_value_heads, seq_len_kv, head_dim,
        dtype=torch.bfloat16, device=device
    )
    
    return {
        "grad_attn_output": grad_attn_output,
        "attn_weights": attn_weights,
        "attn_weights_dropped": attn_weights_dropped,
        "value_states": value_states,
        "dropout_mask": dropout_mask,
        "attention_dropout": attention_dropout,
    }


# ---------------------------------------------------------------------------
# Optimized backward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def run(
    grad_attn_output: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_weights_dropped: torch.Tensor,
    value_states: torch.Tensor,
    dropout_mask: torch.Tensor,
    attention_dropout: float,
):
    """Optimized GQA attention backward.
    
    Key optimizations:
    1. bf16 matmuls (full B200 tensor core throughput, f32 accumulator in cuBLAS)
    2. GQA-aware reshape: no V expansion for matmul 1, implicit aggregation for matmul 2
    3. torch.compile(dynamic=True) fused element-wise: dropout_bwd + softmax_bwd
    4. Layout optimization: (B, H, Sq, D) input avoids transpose
    """
    num_key_value_heads = 8
    num_key_value_groups = 10
    
    batch_size = grad_attn_output.shape[0]
    head_dim = value_states.shape[3]
    seq_len_kv = value_states.shape[2]
    
    # ── Handle input layout ─────────────────────────────────────────────
    if grad_attn_output.shape[1] != 80:
        grad_attn_output = grad_attn_output.transpose(1, 2).contiguous()
    
    seq_len_q = grad_attn_output.shape[2]
    GSq = num_key_value_groups * seq_len_q
    
    # ── GQA-grouped view (zero-cost) ────────────────────────────────────
    go_gqa = grad_attn_output.reshape(batch_size, num_key_value_heads, GSq, head_dim)
    
    # ── Matmul 1: bf16, GQA-aware (no V expansion) ─────────────────────
    # (B, Hkv, G*Sq, D) @ (B, Hkv, D, Skv) → (B, Hkv, G*Sq, Skv) → (B, H, Sq, Skv)
    grad_attn_weights_dropped = torch.matmul(go_gqa, value_states.transpose(-2, -1))
    grad_attn_weights_dropped = grad_attn_weights_dropped.view(
        batch_size, num_key_value_heads * num_key_value_groups, seq_len_q, seq_len_kv
    )
    
    # ── Fused dropout_bwd + softmax_bwd (compiled, single pass) ─────────
    if attention_dropout > 0.0:
        grad_attn_scores = _fused_dropout_softmax_bwd(
            grad_attn_weights_dropped, attn_weights, dropout_mask,
            1.0 / (1.0 - attention_dropout),
        )
    else:
        grad_attn_scores = _fused_softmax_bwd_no_dropout(
            grad_attn_weights_dropped, attn_weights,
        )
    
    # ── Matmul 2: bf16, implicit GQA aggregation ────────────────────────
    # (B, Hkv, Skv, G*Sq) @ (B, Hkv, G*Sq, D) → (B, Hkv, Skv, D)
    aw_dropped_gqa = attn_weights_dropped.view(
        batch_size, num_key_value_heads, GSq, seq_len_kv
    )
    grad_value_states = torch.matmul(aw_dropped_gqa.transpose(-2, -1), go_gqa)
    
    return grad_attn_scores, grad_value_states
