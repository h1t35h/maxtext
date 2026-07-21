import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np

def gated_delta_rule_fwd_kernel(
    q_ref, k_ref, u_ref, w_ref, g_ref, h_init_ref, mask_ref,
    o_ref, h_intermediates_ref
):
    num_chunks = q_ref.shape[2]
    chunk_size = q_ref.shape[3]
    k_dim = q_ref.shape[4]
    v_dim = u_ref.shape[4]
    
    # Initialize hidden state in SRAM
    h = jnp.zeros((k_dim, v_dim), dtype=jnp.float32)
    h = h + h_init_ref[0, 0, :, :].astype(jnp.float32)
    
    def loop_body(chunk_idx, h):
        # Save intermediate
        h_intermediates_ref[0, 0, chunk_idx, :, :] = h.astype(h_intermediates_ref.dtype)
        
        # Load inputs for this chunk
        q = q_ref[0, 0, chunk_idx, :, :]
        k = k_ref[0, 0, chunk_idx, :, :]
        u = u_ref[0, 0, chunk_idx, :, :]
        w = w_ref[0, 0, chunk_idx, :, :]
        g = g_ref[0, 0, chunk_idx, :]
        
        # Computation
        q_g = q.astype(jnp.float32) * jnp.exp(g)[..., None]
        attn_inter = jnp.dot(q_g, h)
        
        v_prime = jnp.dot(w.astype(jnp.float32), h)
        v_new = u.astype(jnp.float32) - v_prime
        
        attn = jnp.dot(q, k.T, preferred_element_type=jnp.float32)
        
        g_diff = g[..., :, None] - g[..., None, :]
        mask_intra = mask_ref[...]
        g_diff = jnp.where(mask_intra, g_diff, -1e30)
        
        attn_i = attn * jnp.exp(g_diff)
        attn_i = jnp.where(mask_intra, attn_i, 0.0)
        
        term2 = jnp.dot(attn_i, v_new)
        o_c = attn_inter + term2
        o_ref[0, 0, chunk_idx, :, :] = o_c.astype(o_ref.dtype)
        
        g_i_last_exp = jnp.exp(g[-1])
        h_new = h * g_i_last_exp
        
        g_diff_exp_state = jnp.exp(g[-1] - g)[..., None]
        k_i_g_diff = k.astype(jnp.float32) * g_diff_exp_state
        update_term = jnp.dot(k_i_g_diff.T, v_new)
        
        h_new = h_new + update_term
        return h_new
        
    # lax.fori_loop handles the recurrence natively in Pallas
    h = lax.fori_loop(0, num_chunks, loop_body, h)

def gated_delta_rule_bwd_kernel(
    q_ref, k_ref, u_ref, w_ref, g_ref, h_intermediates_ref, do_ref, mask_ref,
    dq_ref, dk_ref, du_ref, dw_ref, dg_ref, dh_init_ref
):
    num_chunks = q_ref.shape[2]
    chunk_size = q_ref.shape[3]
    k_dim = q_ref.shape[4]
    v_dim = u_ref.shape[4]
    
    dh = jnp.zeros((k_dim, v_dim), dtype=jnp.float32)
    
    def loop_body(i, dh):
        chunk_idx = num_chunks - 1 - i
        
        # Load inputs
        q = q_ref[0, 0, chunk_idx, :, :]
        k = k_ref[0, 0, chunk_idx, :, :]
        u = u_ref[0, 0, chunk_idx, :, :]
        w = w_ref[0, 0, chunk_idx, :, :]
        g = g_ref[0, 0, chunk_idx, :]
        h = h_intermediates_ref[0, 0, chunk_idx, :, :].astype(jnp.float32)
        do_c = do_ref[0, 0, chunk_idx, :, :].astype(jnp.float32)
        
        # Recompute forward pass intermediates
        q_g = q.astype(jnp.float32) * jnp.exp(g)[..., None]
        v_prime = jnp.dot(w.astype(jnp.float32), h)
        v_new = u.astype(jnp.float32) - v_prime
        attn = jnp.dot(q, k.T, preferred_element_type=jnp.float32)
        g_diff = g[..., :, None] - g[..., None, :]
        mask_intra = mask_ref[...]
        g_diff = jnp.where(mask_intra, g_diff, -1e30)
        attn_i = jnp.where(mask_intra, attn * jnp.exp(g_diff), 0.0)
        k_i_g_diff = k.astype(jnp.float32) * jnp.exp(g[-1] - g)[..., None]
        
        # Backward pass
        d_attn_i = jnp.where(mask_intra, jnp.dot(do_c, v_new.T), 0.0)
        d_v_new_do = jnp.dot(attn_i.T, do_c)
        d_k_i_g_diff = jnp.dot(v_new, dh.T)
        d_v_new_dh = jnp.dot(k_i_g_diff, dh)
        
        d_v_new = d_v_new_do + d_v_new_dh
        du = d_v_new
        dw = -jnp.dot(d_v_new, h.T)
        
        d_attn = d_attn_i * jnp.exp(g_diff)
        dq_attn = jnp.dot(d_attn, k)
        dk_attn = jnp.dot(d_attn.T, q)
        
        d_q_g = jnp.dot(do_c, h.T)
        
        dq = dq_attn + d_q_g * jnp.exp(g)[..., None]
        dk = dk_attn + d_k_i_g_diff * jnp.exp(g[-1] - g)[..., None]
        
        M = d_attn_i * attn_i
        S = jnp.sum(d_k_i_g_diff * k_i_g_diff, axis=1)
        
        dg = jnp.sum(d_q_g * q_g, axis=1) + jnp.sum(M, axis=1) - jnp.sum(M, axis=0) - S
        dg = dg.at[-1].add(jnp.sum(S) + jnp.sum(dh * h) * jnp.exp(g[-1]))
        
        dh_prev = dh * jnp.exp(g[-1]) - jnp.dot(w.astype(jnp.float32).T, d_v_new) + jnp.dot(q_g.T, do_c)
        
        # Write gradients for this chunk
        dq_ref[0, 0, chunk_idx, :, :] = dq.astype(dq_ref.dtype)
        dk_ref[0, 0, chunk_idx, :, :] = dk.astype(dk_ref.dtype)
        du_ref[0, 0, chunk_idx, :, :] = du.astype(du_ref.dtype)
        dw_ref[0, 0, chunk_idx, :, :] = dw.astype(dw_ref.dtype)
        dg_ref[0, 0, chunk_idx, :] = dg.astype(dg_ref.dtype)
        
        return dh_prev
        
    dh = lax.fori_loop(0, num_chunks, loop_body, dh)
    dh_init_ref[0, 0, :, :] = dh.astype(dh_init_ref.dtype)

def gated_delta_rule_fwd(q, k, u, w, g, h_init):
    B, H, num_chunks, chunk_size, K_dim = q.shape
    V_dim = u.shape[-1]
    mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    
    out_shape = jax.ShapeDtypeStruct((B, H, num_chunks, chunk_size, V_dim), q.dtype)
    h_intermediates_shape = jax.ShapeDtypeStruct((B, H, num_chunks, K_dim, V_dim), q.dtype)
    
    in_specs = [
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, chunk_size, K_dim)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, chunk_size, K_dim)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, chunk_size, V_dim)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, chunk_size, K_dim)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0), block_shape=(1, 1, num_chunks, chunk_size)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0), block_shape=(1, 1, K_dim, V_dim)),
        pl.BlockSpec(index_map=lambda b, h: (0, 0), block_shape=(chunk_size, chunk_size)),
    ]
    out_specs = [
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, chunk_size, V_dim)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, K_dim, V_dim)),
    ]
    
    o, h_intermediates = pl.pallas_call(
        gated_delta_rule_fwd_kernel,
        out_shape=(out_shape, h_intermediates_shape),
        grid=(B, H),
        in_specs=in_specs,
        out_specs=out_specs
    )(q, k, u, w, g, h_init, mask)
    
    return o, h_intermediates

def gated_delta_rule_fwd_vjp(q, k, u, w, g, h_init):
    o, h_intermediates = gated_delta_rule_fwd(q, k, u, w, g, h_init)
    return o, (q, k, u, w, g, h_intermediates)

def gated_delta_rule_bwd_vjp(res, do):
    q, k, u, w, g, h_intermediates = res
    B, H, num_chunks, chunk_size, K_dim = q.shape
    V_dim = u.shape[-1]
    mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    
    out_shapes = (
        jax.ShapeDtypeStruct(q.shape, q.dtype), # dq
        jax.ShapeDtypeStruct(k.shape, k.dtype), # dk
        jax.ShapeDtypeStruct(u.shape, u.dtype), # du
        jax.ShapeDtypeStruct(w.shape, w.dtype), # dw
        jax.ShapeDtypeStruct(g.shape, g.dtype), # dg
        jax.ShapeDtypeStruct((B, H, K_dim, V_dim), q.dtype), # dh_init
    )
    
    in_specs = [
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, chunk_size, K_dim)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, chunk_size, K_dim)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, chunk_size, V_dim)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, chunk_size, K_dim)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0), block_shape=(1, 1, num_chunks, chunk_size)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, K_dim, V_dim)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, chunk_size, V_dim)),
        pl.BlockSpec(index_map=lambda b, h: (0, 0), block_shape=(chunk_size, chunk_size)),
    ]
    
    out_specs = [
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, chunk_size, K_dim)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, chunk_size, K_dim)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, chunk_size, V_dim)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0, 0), block_shape=(1, 1, num_chunks, chunk_size, K_dim)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0), block_shape=(1, 1, num_chunks, chunk_size)),
        pl.BlockSpec(index_map=lambda b, h: (b, h, 0, 0), block_shape=(1, 1, K_dim, V_dim)),
    ]
    
    dq, dk, du, dw, dg, dh_init = pl.pallas_call(
        gated_delta_rule_bwd_kernel,
        out_shape=out_shapes,
        grid=(B, H),
        in_specs=in_specs,
        out_specs=out_specs
    )(q, k, u, w, g, h_intermediates, do, mask)
    
    return dq, dk, du, dw, dg, dh_init

@jax.custom_vjp
def gated_delta_rule(q, k, u, w, g, h_init):
    o, _ = gated_delta_rule_fwd(q, k, u, w, g, h_init)
    return o

gated_delta_rule.defvjp(gated_delta_rule_fwd_vjp, gated_delta_rule_bwd_vjp)