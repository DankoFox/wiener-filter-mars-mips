import numpy as np
import os
from typing import List, Tuple
import matplotlib.pyplot as plt

# ---------- Functions implementing the math ----------


def read_sequence_from_file(path: str) -> List[float]:
    """Read whitespace/newline-separated floating numbers from file."""
    with open(path, "r") as f:
        tokens = f.read().strip().split()
    return [float(t) for t in tokens]


def estimate_autocorrelation(
    x: np.ndarray, maxlag: int, biased: bool = True
) -> np.ndarray:
    """Estimate autocorrelation gamma_xx(k) for k = -maxlag..maxlag (but we'll return k=0..maxlag).
    Biased estimator uses denominator N; unbiased would divide by (N-|k|).
    We'll compute gamma_xx(k) = (1/N) * sum_{n=0}^{N-1} x[n]*x[n-k] with zero for out-of-range indices.
    """
    N = len(x)
    gam = np.zeros(maxlag + 1, dtype=float)

    for k in range(maxlag + 1):
        s = 0.0

        for n in range(N):
            idx = n - k
            if 0 <= idx < N:
                s += x[n] * x[idx]

        denom = N if biased else max(1, N - k)
        gam[k] = s / denom
    return gam


def estimate_crosscorrelation(
    d: np.ndarray, x: np.ndarray, maxlag: int, biased: bool = True
) -> np.ndarray:
    """Estimate gamma_dx(l) for l = 0..maxlag.
    gamma_dx(l) = (1/N) * sum_n d[n] * x[n - l] (zero when out-of-range)
    """
    N = len(x)
    gam = np.zeros(maxlag + 1, dtype=float)

    for l in range(maxlag + 1):
        s = 0.0
        for n in range(N):
            idx = n - l
            if 0 <= idx < N:
                s += d[n] * x[idx]

        denom = N if biased else max(1, N - l)
        gam[l] = s / denom
    return gam


def build_R_from_gamma_xx(gamma_xx: np.ndarray, M: int) -> np.ndarray:
    """Build the MxM Toeplitz autocorrelation matrix R where R[l,k] = gamma_xx(l-k).
    For negative lag, we use gamma_xx(|lag|) because autocorrelation is conjugate symmetric;
    but since sequences are real, gamma_xx(-m) = gamma_xx(m).
    gamma_xx array is provided for k=0..maxlag (maxlag >= M-1).
    """
    R = np.zeros((M, M), dtype=float)

    for l in range(M):
        for k in range(M):
            lag = abs(l - k)

            if lag < len(gamma_xx):
                R[l, k] = gamma_xx[lag]
            else:
                R[l, k] = 0.0

    return R


def compute_hopt_123(R: np.ndarray, gamma_d: np.ndarray) -> np.ndarray:
    """Solve R h = gamma_d for hopt. Use pseudo-inverse if R is singular or ill-conditioned."""
    try:
        # Prefer solve for numerical stability
        h = np.linalg.solve(R, gamma_d)
    except np.linalg.LinAlgError:
        h = np.linalg.pinv(R).dot(gamma_d)


def compute_hopt(R, b):
    """
    Solve R x = b where R is symmetric positive-definite (SPD).
    Implementation uses:
      1) Cholesky factorization R = L L^T (L lower triangular)
      2) Forward-substitution: L y = b
      3) Backward-substitution: L^T x = y
    """
    import math
    import numpy as np

    R = np.asarray(R, dtype=float)
    b = np.asarray(b, dtype=float)
    n = R.shape[0]

    # 1) Factorization: compute L such that R = L L^T
    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):  # j = 0..i
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                # diagonal element
                val = R[i, i] - s
                if val <= 0.0:
                    raise ValueError(
                        "Matrix not positive-definite (non-positive pivot)"
                    )
                L[i, j] = math.sqrt(val)
            else:
                # off-diagonal element
                L[i, j] = (R[i, j] - s) / L[j, j]

    # 2) Forward substitution: L temp = b
    temp = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        for k in range(i):
            s += L[i, k] * temp[k]
        temp[i] = (b[i] - s) / L[i, i]

    # 3) Backward substitution: L^T x = y
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for k in range(i + 1, n):
            s += L[k, i] * x[k]
        x[i] = (temp[i] - s) / L[i, i]

    return x


def filter_signal(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply FIR filter h to x with zero-padding for indices < 0 (causal FIR).
    y[n] = sum_{k=0..M-1} h[k] * x[n - k], with x[idx<0] = 0.
    """
    N = len(x)
    M = len(h)
    y = np.zeros(N, dtype=float)

    for n in range(N):
        s = 0.0
        for k in range(M):
            idx = n - k
            if 0 <= idx < N:
                s += h[k] * x[idx]
        y[n] = s

    return y


def compute_mmse(d: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between desired d and output y (average over samples)."""
    diff = d - y
    return float(np.mean(diff * diff))


# ---------- Main pipeline that follows assignment math ----------


def wiener_filter_from_sequences(
    desired: List[float], input_sig: List[float], M: int = 3
) -> Tuple[List[float], float, np.ndarray]:
    """
    Given desired and input sequences (lists of same length N), estimate Wiener FIR filter of length M,
    return (output_sequence, mmse, hopt_matrix_like).
    The returned optimize_coefficient will be a column vector shape (M,).
    """
    if len(desired) != len(input_sig):
        raise ValueError("Error: size not match")
    N = len(input_sig)
    d = np.array(desired, dtype=float)
    x = np.array(input_sig, dtype=float)

    # Estimate autocorrelation up to lag M-1
    gamma_xx = estimate_autocorrelation(x, maxlag=M - 1, biased=True)

    # Cross-correlation vector gamma_dx(l) for l=0..M-1
    gamma_dx = estimate_crosscorrelation(d, x, maxlag=M - 1, biased=True)

    # Build R matrix and solve
    R = build_R_from_gamma_xx(gamma_xx, M)
    hopt = compute_hopt(R, gamma_dx)
    # Filter signal
    y = filter_signal(x, hopt)
    mmse = compute_mmse(d, y)
    return list(y), mmse, hopt


# ---------- Benchmarking ----------


def find_best_filter_length(desired_seq, input_seq, M_min=1, M_max=10):
    """
    Try different filter lengths M between M_min and M_max (inclusive)
    and return the one that gives the minimum MMSE.
    """
    results = []

    for M in range(M_min, M_max + 1):
        try:
            _, mmse_val, _ = wiener_filter_from_sequences(desired_seq, input_seq, M)
            results.append((M, mmse_val))
            print(f"M={M:2d} → MMSE={mmse_val:.8f}")
        except np.linalg.LinAlgError:
            results.append((M, np.inf))
            print(f"M={M:2d} → Singular R (skipped)")

    best_M, best_mmse = min(results, key=lambda x: x[1])
    print(f"\n✅ Best M = {best_M} with MMSE = {best_mmse:.8f}")
    return best_M, best_mmse, results


# ---------- I/O and CLI-style behavior (demonstration) ----------


def format_output_lines(output_seq: List[float], mmse: float) -> Tuple[str, str]:
    """Format two lines: 1) output sequence (space-separated, one decimal) 2) MMSE value (6 decimals)."""
    seq_line = " ".join(f"{v:.1f}" for v in output_seq)
    mmse_line = f"{mmse:.6f}"
    return seq_line, mmse_line


def write_output_file(path: str, seq_line: str, mmse_line: str):
    with open(path, "w") as f:
        f.write(seq_line + "\n")
        f.write(mmse_line + "\n")


def plot_mmse_vs_M(results):
    Ms = [r[0] for r in results]
    MMSEs = [r[1] for r in results]
    plt.plot(Ms, MMSEs, marker="o")
    plt.xlabel("Filter length M")
    plt.ylabel("MMSE")
    plt.title("Wiener Filter Length Optimization")
    plt.grid(True)
    plt.show()


def main(
    desired_path="desired.txt",
    input_path="input.txt",
    output_path="output.txt",
    M=3,
    demo_create_files=True,
    verbose=True,
):
    """
    If demo_create_files True and files do not exist, create example desired.txt and input.txt for demonstration.
    Verbose=True will print detailed intermediate values for debugging and verification.
    """
    if True:
        desired_sample = [100.0, -1.2, 10.9, 1.1, 2435.3, 11.0, -0.8, 66.0, -0.9, 1.1]
        noise = np.random.normal(0, 0.05 * 10, 10)

        # input_sample = [d+p for d,p in zip(desired_sample,noise)]
        input_sample = [99.5, -1.2, 10.4, 1.2, 2434.5, 11.5, -0.5, 67.0, -1.5, 1.2]

        with open(desired_path, "w") as f:
            f.write(" ".join(f"{v:.1f}" for v in desired_sample) + "\n")
        with open(input_path, "w") as f:
            f.write(" ".join(f"{v:.1f}" for v in input_sample) + "\n")
        print(f"✅ Demo files created: {desired_path}, {input_path}")

    try:
        desired_seq = read_sequence_from_file(desired_path)
        input_seq = read_sequence_from_file(input_path)
    except FileNotFoundError as e:
        print(
            "❌ Error: input files not found. Please place desired.txt and input.txt in the working directory."
        )
        raise e

    if len(desired_seq) != len(input_seq):
        err = "Error: size not match"
        print(err)
        write_output_file(output_path, err, "")
        return

    # Compute Wiener filter results
    output_seq, mmse_val, hopt = wiener_filter_from_sequences(
        desired_seq, input_seq, M=M
    )

    # Expose variables for grading
    global desired_signal, input_signal, optimize_coefficient, mmse, output_signal
    desired_signal = desired_seq
    input_signal = input_seq
    optimize_coefficient = hopt
    mmse = mmse_val
    output_signal = output_seq

    # ---------- Print detailed results ----------
    if verbose:
        np.set_printoptions(suppress=True)
        print("\n=== Wiener Filter Computation Details ===")
        print(f"Filter length (M): {M}")
        print(f"Input signal (x):      {np.round(input_signal, 3)}")
        print(f"Desired signal (d):    {np.round(desired_signal, 3)}")

        # Recompute for details
        gamma_xx = estimate_autocorrelation(
            np.array(input_signal), maxlag=M - 1, biased=True
        )
        gamma_dx = estimate_crosscorrelation(
            np.array(desired_signal), np.array(input_signal), maxlag=M - 1, biased=True
        )
        R = build_R_from_gamma_xx(gamma_xx, M)

        print("\nAutocorrelation γxx (0..M-1):", np.round(gamma_xx, 6))
        print("Cross-correlation γdx (0..M-1):", np.round(gamma_dx, 6))
        print("\nR (Toeplitz matrix):\n", np.round(R, 6))
        print("\nOptimal coefficients hopt:", np.round(hopt, 6))
        print("Output signal (filtered):", np.round(output_seq, 4))
        print(f"MMSE: {mmse_val:.6f}")
        print("==========================================\n")

        print("\n--- Searching for best filter length ---")
        best_M, best_mmse, results = find_best_filter_length(
            desired_seq, input_seq, M_min=1, M_max=10
        )
        plot_mmse_vs_M(results)

    seq_line, mmse_line = format_output_lines(output_seq, mmse_val)
    write_output_file(output_path, seq_line, mmse_line)
    print(f"✅ Results written to {output_path}")


# Run demonstration
if __name__ == "__main__":
    main(
        desired_path="./desired.txt",
        input_path="./input.txt",
        output_path="./output.txt",
        M=7,
        demo_create_files=True,
    )
