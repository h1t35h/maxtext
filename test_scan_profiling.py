import jax
import jax.numpy as jnp
from flax import nnx
import os
import contextlib

@contextlib.contextmanager
def profile(log_dir="/tmp/test_scan_profiles"):
    with jax.profiler.trace(log_dir):
        yield

class DecoderLayer(nnx.Module):
    def __init__(self, dim, rngs):
        self.linear1 = nnx.Linear(dim, dim, rngs=rngs)
        self.linear2 = nnx.Linear(dim, dim, rngs=rngs)
    
    def __call__(self, x):
        # Adding a named scope here to see if it shows up in the profile
        with jax.named_scope("decoder_layer_internals"):
            x = self.linear1(x)
            x = jax.nn.relu(x)
            x = self.linear2(x)
        return x

class StackedDecoderLayers(nnx.Module):
    def __init__(self, num_layers, dim, rngs):
        self.num_layers = num_layers
        forked_rngs = rngs.fork(split=num_layers)
        self.layers = nnx.vmap(
            lambda r: DecoderLayer(dim, r),
            in_axes=0, out_axes=0,
            axis_size=num_layers,
        )(forked_rngs)

    def __call__(self, x):
        # To scan over the layers, we split the state
        graphdef, params, state = nnx.split(self.layers, nnx.Param, ...)
        
        def scan_fn(carry, scanned_vars):
            # jax.named_scope allows injecting names into XLA/TensorBoard profiler traces
            with jax.named_scope("scan_fn_body"):
                current_params, current_state = scanned_vars
                # Merge the state back for the current slice
                layer = nnx.merge(graphdef, current_params, current_state)
                carry = layer(carry)
                return carry, nnx.state(layer)
        
        # Adding a named call for the scan itself
        @jax.named_call
        def do_scan(carry, params, state):
            return jax.lax.scan(scan_fn, carry, (params, state))

        # We execute the scan
        x, out_state = do_scan(x, params, state)
        
        # update self.layers state
        nnx.update(self.layers, out_state)
        return x

if __name__ == "__main__":
    dim = 128
    num_layers = 4
    batch_size = 2
    
    model = StackedDecoderLayers(num_layers, dim, rngs=nnx.Rngs(0))
    x = jnp.ones((batch_size, dim))
    
    @jax.named_call
    def run_model(model, x):
        return model(x)

    @jax.jit
    def run_jit(model, x):
        return run_model(model, x)
        
    print("Warming up JIT...")
    run_jit(model, x).block_until_ready()
    
    print("Profiling...")
    # Add named_scope in the outer eager context just in case
    with jax.named_scope("profiling_session"):
        with profile():
            run_jit(model, x).block_until_ready()
        
    print("Done! You can view the profile with:")
    print("tensorboard --logdir=/tmp/test_scan_profiles")
