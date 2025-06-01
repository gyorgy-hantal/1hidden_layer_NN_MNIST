import numpy as np
import warp as wp
import time  # Use time.perf_counter for more precise timing
import torch

# Initialize Warp (if not done elsewhere, good practice to have it explicitly)
# wp.init() # Not strictly needed if other wp calls trigger it, but good for clarity.

# tile size (for tiled_gemm)
TILE_M = wp.constant(64)
TILE_N = wp.constant(64)
TILE_K = wp.constant(64)

# num threads per-tile (for tiled_gemm)
TILE_THREADS = 128  # This translates to block_dim for launch_tiled


@wp.kernel
def tile_gemm(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):
    # output tile index
    i, j = wp.tid()  # i is block_idx_x (for M dimension), j is block_idx_y (for N dimension)

    # sum tile is TILE_M x TILE_N
    # This tile will reside in registers or be spilled if too large,
    # wp.tile_zeros implies it's a tile structure managed by Warp.
    sum_tile_data = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

    M_dim = A.shape[0]
    N_dim = B.shape[1]
    K_dim = A.shape[1]

    # Number of inner tiles along K dimension
    num_k_tiles = int(K_dim / TILE_K)  # Assuming K_dim is perfectly divisible by TILE_K

    for k_tile_idx in range(0, num_k_tiles):
        # Load TILE_M x TILE_K tile from A
        # offset is (row_offset, col_offset)
        # i is the tile row index, k_tile_idx is the tile col index for A
        a_sub_tile = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k_tile_idx * TILE_K))

        # Load TILE_K x TILE_N tile from B
        # k_tile_idx is the tile row index, j is the tile col index for B
        b_sub_tile = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k_tile_idx * TILE_K, j * TILE_N))

        # Perform tile matrix multiplication: sum_tile_data += a_sub_tile @ b_sub_tile
        wp.tile_matmul(a_sub_tile, b_sub_tile, sum_tile_data)

    # Store the resulting TILE_M x TILE_N sum_tile_data to C
    wp.tile_store(C, sum_tile_data, offset=(i * TILE_M, j * TILE_N))


@wp.kernel
def global_gemm(A: wp.array2d(dtype=float),
                B: wp.array2d(dtype=float),
                C: wp.array2d(dtype=float),
                M_dim: int,  # Rows of C (and A)
                N_dim: int,  # Cols of C (and B)
                K_dim: int  # Cols of A (and rows of B)
                ):
    # Global thread ID, corresponds to an element in C
    r, c = wp.tid()  # r: row of C, c: col of C

    # Boundary check (optional if launch dimensions are exact, good practice otherwise)
    if r >= M_dim or c >= N_dim:
        return

    sum_val = float(0.0)  # Initialize sum for C[r,c]
    for k_idx in range(K_dim):
        sum_val += A[r, k_idx] * B[k_idx, c]

    C[r, c] = sum_val


if __name__ == "__main__":
    wp.init()  # Explicitly initialize Warp
    device = wp.get_preferred_device()  # Use the preferred device (CPU or GPU)
    print(f"Running on device: {device}")

    # generate some tile aligned matrix dimensions
    # For simplicity in the kernels, M, K, N are assumed to be divisible by TILE_M, TILE_K, TILE_N respectively
    m_factor = 64 # Increase factors for larger matrices to see more performance difference
    k_factor = 128
    n_factor = 64

    M = TILE_M * m_factor
    K = TILE_K * k_factor
    N = TILE_N * n_factor

    print(f"Matrix dimensions: M={M}, K={K}, N={N}")

    rng = np.random.default_rng(42)
    A_np = rng.random((M, K), dtype=np.float32)
    B_np = rng.random((K, N), dtype=np.float32)

    # --- Tiled GEMM ---
    print("\n--- Running Tiled GEMM ---")
    A_wp = wp.array(A_np, device=device)  # Ensure arrays are on the chosen device
    B_wp = wp.array(B_np, device=device)
    C_tiled_wp = wp.zeros(shape=(M, N), device=device)  # Output array for tiled version

    # Warm-up for JIT compilation (optional, but good for fair timing)
    wp.launch_tiled(
        tile_gemm,
        dim=(int(M / TILE_M), int(N / TILE_N)),  # Number of output tiles
        inputs=[A_wp, B_wp, C_tiled_wp],
        block_dim=TILE_THREADS,  # Threads cooperating on one tile operation
        device=device
    )
    wp.synchronize()

    # Timed run

    start_time_tiled = time.perf_counter()
    wp.launch_tiled(
        tile_gemm,
        dim=(int(M / TILE_M), int(N / TILE_N)),
        inputs=[A_wp, B_wp, C_tiled_wp],
        block_dim=TILE_THREADS,
        device=device
    )
    # If just timing execution, tape. όχι necessary. `wp.synchronize()` after launch is key.
    wp.synchronize()  # Ensure kernel finishes
    end_time_tiled = time.perf_counter()
    tiled_gemm_time = end_time_tiled - start_time_tiled

    C_tiled_np = C_tiled_wp.numpy()
    # Correctness check (NumPy's @ is highly optimized)
    # Using a larger tolerance for GPU float arithmetic
    assert (np.allclose(C_tiled_np, A_np @ B_np, atol=1e-4, rtol=1e-3)), "Tiled GEMM failed correctness check"
    print(f"Tiled GEMM time: {tiled_gemm_time:.6f} seconds")
    print("Tiled matrix multiplication passed correctness check.")

    # --- Global Memory GEMM ---
    print("\n--- Running Global Memory GEMM ---")
    # A_wp and B_wp are already on the device
    C_global_wp = wp.zeros(shape=(M, N), device=device)  # Output array for global version

    # Warm-up for JIT compilation
    wp.launch(
        global_gemm,
        dim=(M, N),  # One thread per output element C[r,c]
        inputs=[A_wp, B_wp, C_global_wp, M, N, K],
        device=device
    )
    wp.synchronize()

    # Timed run
    start_time_global = time.perf_counter()
    wp.launch(
        global_gemm,
        dim=(M, N),
        inputs=[A_wp, B_wp, C_global_wp, M, N, K],
        device=device
    )
    wp.synchronize()  # Ensure kernel finishes
    end_time_global = time.perf_counter()
    global_gemm_time = end_time_global - start_time_global

    C_global_np = C_global_wp.numpy()
    assert (np.allclose(C_global_np, A_np @ B_np, atol=1e-4, rtol=1e-3)), "Global GEMM failed correctness check"
    print(f"Global memory GEMM time: {global_gemm_time:.6f} seconds")
    print("Global matrix multiplication passed correctness check.")



    # --- PyTorch GEMM ---
    print("\n--- Running PyTorch GEMM ---")

    A_torch = wp.to_torch(A_wp)
    B_torch = wp.to_torch(B_wp)
    C_torch = torch.zeros((M, N), dtype=torch.float32).to(torch.accelerator.current_accelerator())


    # Warm-up for PyTorch (e.g., to load cuBLAS kernels if on GPU)
    _ = torch.matmul(A_torch, B_torch, out=C_torch)
    torch.cuda.synchronize(device=device)  # Synchronize on the specific torch device

    # Timed run for PyTorch
    start_time_torch = time.perf_counter()
    torch.matmul(A_torch, B_torch, out=C_torch)
    torch.cuda.synchronize(device=device)  # Ensure CUDA operations complete before stopping timer
    end_time_torch = time.perf_counter()
    torch_gemm_time = end_time_torch - start_time_torch

    # Correctness check for PyTorch result
    C_torch_np = C_torch.cpu().numpy()  # Move to CPU before converting to NumPy for comparison
    assert (np.allclose(C_torch_np, A_np @ B_np, atol=1e-4, rtol=1e-3)), "PyTorch GEMM failed correctness check"
    print(f"PyTorch GEMM time: {torch_gemm_time:.6f} seconds")
    print("PyTorch matrix multiplication passed correctness check.")

    # --- Updated Performance Summary ---
    # Replace your previous summary section with this or modify it:
    print("\n--- Extended Performance Summary ---")
    print(f"Matrix dimensions: M={M}, K={K}, N={N} (on device: {device})")
    print(
        f"Tile sizes: M={TILE_M}, N={TILE_N}, K={TILE_K} with {TILE_THREADS} threads/block for Tiled GEMM.")
    print(f"Tiled GEMM (Warp) time: {tiled_gemm_time:.6f} seconds")
    print(f"Global GEMM (Warp) time: {global_gemm_time:.6f} seconds")
    print(f"PyTorch GEMM time: {torch_gemm_time:.6f} seconds")

    # Comparisons
    if tiled_gemm_time > 0 and torch_gemm_time > 0:
        if tiled_gemm_time < torch_gemm_time:
            print(f"  - Tiled GEMM (Warp) is {torch_gemm_time / tiled_gemm_time:.2f}x faster than PyTorch GEMM.")
        else:
            print(f"  - PyTorch GEMM is {tiled_gemm_time / torch_gemm_time:.2f}x faster than Tiled GEMM (Warp).")

    if global_gemm_time > 0 and torch_gemm_time > 0:
        if global_gemm_time < torch_gemm_time:
            print(f"  - Global GEMM (Warp) is {torch_gemm_time / global_gemm_time:.2f}x faster than PyTorch GEMM.")
        else:
            print(f"  - PyTorch GEMM is {global_gemm_time / torch_gemm_time:.2f}x faster than Global GEMM (Warp).")

    if tiled_gemm_time > 0 and global_gemm_time > 0:
        if tiled_gemm_time < global_gemm_time:
            print(f"  - Tiled GEMM (Warp) is {global_gemm_time / tiled_gemm_time:.2f}x faster than Global GEMM (Warp).")
        else:
            print(
                f"  - Global GEMM (Warp) is {tiled_gemm_time / global_gemm_time:.2f}x faster than Tiled GEMM (Warp) (or similar).")
