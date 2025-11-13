import subprocess
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("TkAgg")

# ==============================
# CONFIGURATION
# ==============================
num_tests = 1000
mars_jar = "Mars4_5.jar"  # path to MARS JAR
asm_file = "main.asm"  # assembly file
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)

noise_types = ["white", "pink", "brown", "multi_tone", "impulse_spikes"]


# ==============================
# SIGNAL GENERATION
# ==============================


def generate_signals(noise_type="white"):
    """
    Generate desired signal (sine wave) and noisy input based on noise_type.
    noise_type: "white", "pink", "brown", "multi_tone", "impulse_spikes", or "none"
    """
    n = np.arange(10)
    freq = 0.05
    amplitude = 1.0
    desired_signal = amplitude * np.sin(2 * np.pi * freq * n)

    # Noise generation
    if noise_type == "white":
        noise = np.random.normal(0, 0.3, len(n))
    elif noise_type == "pink":
        uneven = len(n) % 2
        X = np.random.randn(len(n) // 2 + 1 + uneven) + 1j * np.random.randn(
            len(n) // 2 + 1 + uneven
        )
        S = np.sqrt(np.arange(len(X)) + 1.0)
        y = np.fft.irfft(X / S).real
        noise = 0.3 * y[: len(n)] / np.std(y)
    elif noise_type == "brown":
        noise = np.cumsum(np.random.normal(0, 0.05, len(n)))
        noise = 0.3 * noise / np.std(noise)
    elif noise_type == "multi_tone":
        freqs = [0.02, 0.07, 0.12]
        amps = [0.4, 0.25, 0.15]
        noise = np.zeros_like(n, dtype=float)
        for f, a in zip(freqs, amps):
            noise += a * np.sin(2 * np.pi * f * n + np.random.uniform(0, 2 * np.pi))
        noise += np.random.normal(0, 0.05, len(n))
        noise = 0.3 * noise / (np.std(noise) + 1e-12)
    elif noise_type == "impulse_spikes":
        noise = np.random.normal(0, 0.05, len(n))
        num_spikes = max(1, len(n) // 10)
        spike_positions = np.random.choice(len(n), num_spikes, replace=False)
        for p in spike_positions:
            noise[p] += np.random.choice([-1.5, 1.5])
    elif noise_type == "none":
        noise = np.zeros_like(n)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # --- Ensure noise has at least 0.1 in magnitude ---
    # This preserves the sign
    noise = np.sign(noise) * np.maximum(np.abs(noise), 0.1)

    noisy_signal = desired_signal + noise

    # Save files as ONE LINE, SPACE-SEPARATED with 1 decimal
    np.savetxt("desired.txt", desired_signal.reshape(1, -1), fmt="%.1f", delimiter=" ")
    np.savetxt("input.txt", noisy_signal.reshape(1, -1), fmt="%.1f", delimiter=" ")

    return desired_signal, noisy_signal


# ==============================
# OUTPUT EXTRACTION
# ==============================
def extract_output():
    try:
        with open("output.txt", "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if len(lines) == 0:
            return None, None

        # Handle error messages like "Error: size not match"
        if lines[0].startswith("Error"):
            print("⚠️ MARS error:", lines[0])
            return None, None

        # Normal case
        filtered_output, mmse = None, None
        for line in lines:
            if line.startswith("Filtered output: "):
                filtered_output = list(
                    map(float, line.replace("Filtered output: ", "").split())
                )
            elif line.startswith("MMSE: "):
                mmse = float(line.replace("MMSE: ", ""))

        if filtered_output is None or mmse is None:
            return None, None

        return filtered_output, mmse
    except Exception as e:
        print(f"⚠️ Error extracting output: {e}")
        return None, None


# ==============================
# RUN TEST
# ==============================
def run_mars(test_num):
    noise_type = noise_types[test_num % len(noise_types)]
    print(f"Running test {test_num + 1}/{num_tests} with noise: {noise_type}")

    generate_signals(noise_type=noise_type)
    command = ["java", "-jar", mars_jar, asm_file]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ MARS execution failed for test {test_num + 1}.")
        return None, None, noise_type

    # Save raw simulator output
    output_filename = os.path.join(output_dir, f"output_{test_num + 1}.txt")
    with open(output_filename, "w") as f:
        f.write(result.stdout)

    filtered_output, mmse = extract_output()
    return filtered_output, mmse, noise_type


# ==============================
# MAIN TEST LOOP
# ==============================
all_mmse, all_filtered_outputs, noise_labels = [], [], []

for i in range(num_tests):
    filtered_output, mmse, noise_type = run_mars(i)

    if filtered_output is not None and mmse is not None:
        print(f"✅ Test {i + 1} done | {noise_type} | MMSE: {mmse:.5f}")
        all_mmse.append(mmse)
        all_filtered_outputs.append(filtered_output)
        noise_labels.append(noise_type)
    else:
        print(f"❌ Test {i + 1} failed.")

print(f"\nFinished {num_tests} tests. Results stored in '{output_dir}'.")

# ==============================
# SUMMARY ANALYSIS
# ==============================
all_mmse = np.array(all_mmse)
all_filtered_outputs = np.array(all_filtered_outputs)
df = pd.DataFrame({"NoiseType": noise_labels, "MMSE": all_mmse})

print("\n=== OVERALL SUMMARY ===")
print(f"Total successful tests: {len(all_mmse)} / {num_tests}")
print(f"Mean MMSE: {np.mean(all_mmse):.4f}")
print(f"Std MMSE:  {np.std(all_mmse):.4f}")
print(f"Min MMSE:  {np.min(all_mmse):.4f}")
print(f"Max MMSE:  {np.max(all_mmse):.4f}")

print("\n=== SUMMARY BY NOISE TYPE ===")
grouped_stats = df.groupby("NoiseType")["MMSE"].agg(
    ["mean", "std", "min", "max", "count"]
)
print(grouped_stats.to_string(float_format=lambda x: f"{x:.4f}"))

# ==============================
# VISUALIZATION
# ==============================
plt.style.use("seaborn-v0_8-darkgrid")

# 1️⃣ MMSE across tests
plt.figure(figsize=(8, 4))
plt.plot(all_mmse, marker="o", linewidth=1)
plt.title("MMSE Across Tests")
plt.xlabel("Test Number")
plt.ylabel("MMSE")
plt.tight_layout()
plt.show()

# 2️⃣ Boxplot per noise type
plt.figure(figsize=(7, 4))
df.boxplot(column="MMSE", by="NoiseType", grid=False)
plt.title("MMSE Distribution by Noise Type")
plt.suptitle("")
plt.tight_layout()
plt.show()

# 3️⃣ Running mean (stability)
running_mean = np.cumsum(all_mmse) / np.arange(1, len(all_mmse) + 1)
plt.figure(figsize=(8, 4))
plt.plot(running_mean, color="tab:blue", linewidth=2)
plt.title("Running Mean of MMSE (Convergence)")
plt.xlabel("Test Number")
plt.ylabel("Mean MMSE")
plt.tight_layout()
plt.show()

# 4️⃣ Signal power vs. MMSE
signal_power = np.mean(np.square(all_filtered_outputs), axis=1)
plt.figure(figsize=(6, 4))
plt.scatter(all_mmse, signal_power, alpha=0.6)
plt.title("Signal Power vs. MMSE")
plt.xlabel("MMSE")
plt.ylabel("Mean Signal Power")
plt.tight_layout()
plt.show()

# 5️⃣ Correlation Heatmap (sample 30)
if len(all_filtered_outputs) >= 30:
    subset = np.random.choice(len(all_filtered_outputs), 30, replace=False)
    corr = np.corrcoef(all_filtered_outputs[subset])
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, cmap="coolwarm", square=True)
    plt.title("Filtered Output Correlation (30 Samples)")
    plt.tight_layout()
    plt.show()

# ==============================
# SAVE RESULTS
# ==============================
df["SignalPower"] = signal_power[: len(df)]
df.to_csv("mmse_results.csv", index=False)
print("✅ Results saved to mmse_results.csv")
