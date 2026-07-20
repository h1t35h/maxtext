import os
import socket
import contextlib
import sys
import jax


@contextlib.contextmanager
def profile(log_dir="/mnt/disks/ps/bg/qwen/profiles"):

    with jax.profiler.trace(log_dir):
        yield

    # 2. extract the last created directory
    plugin_dir = os.path.join(log_dir, "plugins", "profile")
    try:
        # List all items in the log directory
        all_items = [os.path.join(plugin_dir, d) for d in os.listdir(plugin_dir)]
        # Filter for directories only
        subdirs = [d for d in all_items if os.path.isdir(d)]

        if subdirs:
            # Sort by creation time and pick the last one
            latest_dir = max(subdirs, key=os.path.getctime)
            run_name = os.path.basename(latest_dir)

            # Get the current hostname (matches the 'hosts' param in your example)
            hostname = socket.gethostname()

            # 3. Construct the URL
            # Base URL structure based on your request
            base = "http://localhost:6006/data/plugin/profile/trace_viewer@"

            # Construct the parameter string
            # Note: The URL format provided uses params in both the path (;) and query (?)
            params = f"run={run_name}&tag=trace_viewer@&hosts={hostname}"
            path_params = f"run={run_name};tag=trace_viewer@;hosts={hostname}"

            url = f"{base};{path_params}?{params}"

            print(f"\n--- Profiling Complete ---")
            print(f"Latest Run: {run_name}")
            print(f"Trace URL:\n{url}\n")
        else:
            print(f"Warning: No subdirectories found in {log_dir}")

    except Exception as e:
        print(f"Error generating trace URL: {e}")


# Ensure we can import maxtext modules.
# Adjust this path if your colab structure is different.

from maxtext.configs import pyconfig
from maxtext.utils.train_utils import setup_train_loop
from maxtext.trainers.pre_train import train as pre_train
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from flax import nnx

sys.path.append("/mnt/disks/ps/bg/qwen/maxtext/src")


def run_profile_pass():
    # 1. Initialize configuration for Qwen3
    # We use qwen3-8b as a base and reduce the layers to fit into memory.
    print("Initializing configuration...")

    config_args = [
        "",  # Dummy script name for arg parser
        "src/maxtext/configs/base.yml",
        "model_name=qwen3-8b",
        "run_name=qwen3_profile",
        "base_output_directory=/tmp/maxtext_output",
        # --- Memory Constraints / Reductions ---
        "base_num_decoder_layers=2",  # Significantly reduce layers
        "per_device_batch_size=1",
        "max_target_length=128",  # Short sequence for quick simulation
        # --- Training params ---
        "dataset_type=synthetic",
        "steps=1",
        "enable_checkpointing=False",
        "override_model_config=True",
        "pure_nnx=True",
        "enable_goodput_recording=False",
        "enable_checkpoint_cloud_logger=False",
        "monitor_goodput=False",
        f"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
    ]

    # Initialize the MaxText config object
    config = pyconfig.initialize(config_args)

    # 2. Setup training loop (initializes model, optimizer, data iterators)
    print("Setting up training loop (initializing model and optimizer)...")
    (
        init_rng,
        checkpoint_manager,
        state_mesh_shardings,
        model,
        mesh,
        learning_rate_schedule,
        data_iterator,
        data_loader,
        rampup_manager,
        eval_data_iterator,
        train_state,
    ) = setup_train_loop(config, recorder=None)

    # Get a batch of synthetic data
    print("Generating synthetic batch...")
    batch = next(data_iterator)

    from maxtext.utils import train_utils, sharding
    from flax.linen import partitioning as nn_partitioning

    jit_model, state_pure = nnx.split(train_state)
    params_shardings, state_mesh_shardings = (
        sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)
    )

    p_train_step, _ = train_utils.jit_train_and_eval_step(
        config,
        jit_model,
        mesh,
        state_pure,
        state_mesh_shardings,
        pre_train.train_step,
        eval_step=None,
        eval_data_iterator=None,
        params_shardings=params_shardings,
    )

    # 3. Warmup (to compile jitted functions before profiling)
    print(
        "Running warmup pass to trigger JIT compilation (this might take a few minutes)..."
    )
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
        new_state, metrics = p_train_step(state_pure, batch)
        # Block until JAX operations complete
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            new_state,
        )

    # 4. Profile a single pass
    print("Running profiled pass...")
    with profile():
        with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            final_state, final_metrics = p_train_step(new_state, batch)
            jax.tree_util.tree_map(
                lambda x: (
                    x.block_until_ready() if hasattr(x, "block_until_ready") else x
                ),
                final_state,
            )

    # Extract and print the loss
    loss = final_metrics.get("scalar", {}).get("learning/loss")
    print(f"Pass completed successfully with loss: {loss}")


if __name__ == "__main__":
    run_profile_pass()
