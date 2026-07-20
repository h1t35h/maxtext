import jax
import jax.numpy as jnp
import time
import sys
import os

# Adjust the path to import maxtext modules correctly if running locally.
# If running in Colab, make sure your python path includes the `src` directory 
# of the maxtext repository, or copy the function directly into your notebook.
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from maxtext.models.qwen3 import jax_chunk_gated_delta_rule

def profile_delta_rule():
    # ---------------------------------------------------------
    # 1. Setup Data Shapes
    # ---------------------------------------------------------
    B = 4
    seq_len = 1024
    H = 16
    K_dim = 128
    V_dim = 128
    chunk_size = 64
    
    print(f"Profiling with B={B}, seq_len={seq_len}, H={H}, K_dim={K_dim}, V_dim={V_dim}, chunk_size={chunk_size}")

    key_rng = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5 = jax.random.split(key_rng, 5)
    
    # Initialize inputs with appropriate dtypes
    query = jax.random.normal(k1, (B, seq_len, H, K_dim), dtype=jnp.bfloat16)
    key_t = jax.random.normal(k2, (B, seq_len, H, K_dim), dtype=jnp.bfloat16)
    value = jax.random.normal(k3, (B, seq_len, H, V_dim), dtype=jnp.bfloat16)
    g = jax.random.normal(k4, (B, seq_len, H), dtype=jnp.bfloat16)
    beta = jax.random.uniform(k5, (B, seq_len, H), dtype=jnp.bfloat16)
    
    # ---------------------------------------------------------
    # 2. Define Forward and Backward Functions
    # ---------------------------------------------------------
    @jax.jit
    def forward_fn(q, k, v, g_val, beta_val):
        out, _ = jax_chunk_gated_delta_rule(
            q, k, v, g_val, beta_val, chunk_size=chunk_size
        )
        return out

    def loss_fn(q, k, v, g_val, beta_val):
        out, _ = jax_chunk_gated_delta_rule(
            q, k, v, g_val, beta_val, chunk_size=chunk_size
        )
        return jnp.sum(out, dtype=jnp.float32)

    @jax.jit
    def fwd_bwd_fn(q, k, v, g_val, beta_val):
        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(
            q, k, v, g_val, beta_val
        )
        return loss, grads

    # ---------------------------------------------------------
    # 3. Profile Forward Pass
    # ---------------------------------------------------------
    print("\nWarming up forward pass...")
    out = forward_fn(query, key_t, value, g, beta)
    out.block_until_ready()
    
    print("Profiling forward pass...")
    start_time = time.time()
    num_iters = 50
    for _ in range(num_iters):
        out = forward_fn(query, key_t, value, g, beta)
        out.block_until_ready()
    fwd_time_ms = ((time.time() - start_time) / num_iters) * 1000
    print(f"Average Forward Time: {fwd_time_ms:.2f} ms")

    # ---------------------------------------------------------
    # 4. Profile Forward + Backward Pass
    # ---------------------------------------------------------
    # Note: jax_chunk_gated_delta_rule doesn't use a custom VJP (jax.custom_vjp),
    # so JAX's standard autodiff traces through the forward pass operations automatically.
    print("\nWarming up forward + backward pass...")
    loss, grads = fwd_bwd_fn(query, key_t, value, g, beta)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), grads)
    
    print("Profiling forward + backward pass...")
    start_time = time.time()
    for _ in range(num_iters):
        loss, grads = fwd_bwd_fn(query, key_t, value, g, beta)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), grads)
    fwd_bwd_time_ms = ((time.time() - start_time) / num_iters) * 1000
    print(f"Average Fwd + Bwd Time: {fwd_bwd_time_ms:.2f} ms")

if __name__ == "__main__":
    profile_delta_rule()
