import subprocess
import os
import numpy as np

# Define the number of iterations
num_tests = 1000

# Path to your MARS JAR and ASM file
mars_jar = "Mars4_5.jar"  # Modify this to the correct path of your Mars JAR file
asm_file = "main.asm"  # Modify this if needed

# Directory to store outputs
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)

all_mmse = []
all_filtered_outputs = []


# Function to generate the desired signal and noisy signal
def generate_signals(noise_type="white"):
    """
    Generate desired signal (sine wave) and noisy input based on noise_type.
    noise_type: "white", "pink", "brown", or "none"
    """
    n = np.arange(10)
    freq = 0.05
    amplitude = 1.0

    # Desired clean sine wave
    desired_signal = amplitude * np.sin(2 * np.pi * freq * n)

    # Select noise type
    if noise_type == "white":
        noise = np.random.normal(0, 0.3, len(n))  # flat spectrum
    elif noise_type == "pink":
        # Generate 1/f pink noise
        uneven = len(n) % 2
        X = np.random.randn(len(n) // 2 + 1 + uneven) + 1j * np.random.randn(
            len(n) // 2 + 1 + uneven
        )
        S = np.sqrt(np.arange(len(X)) + 1.0)  # 1/f amplitude scaling
        y = np.fft.irfft(X / S).real
        noise = y[: len(n)]
        noise = 0.3 * noise / np.std(noise)
    elif noise_type == "brown":
        # Brownian noise (1/f^2)
        noise = np.cumsum(np.random.normal(0, 0.05, len(n)))
        noise = 0.3 * noise / np.std(noise)
    elif noise_type == "none":
        noise = np.zeros_like(n)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    noisy_signal = desired_signal + noise

    # Save files as ONE LINE, SPACE-SEPARATED
    np.savetxt("desired.txt", desired_signal.reshape(1, -1), fmt="%.1f", delimiter=" ")
    np.savetxt("input.txt", noisy_signal.reshape(1, -1), fmt="%.1f", delimiter=" ")

    return desired_signal, noisy_signal


# Function to extract the filtered output and MMSE from output.txt
def extract_output():
    try:
        with open("output.txt", "r") as f:
            lines = f.readlines()
            # Check if the file has enough lines and contains the correct labels
            if len(lines) >= 2:
                filtered_line = lines[0].strip()
                mmse_line = lines[1].strip()

                # Extract filtered output
                if filtered_line.startswith("Filtered output: "):
                    filtered_output = list(
                        map(
                            float,
                            filtered_line.replace("Filtered output: ", "").split(),
                        )
                    )
                else:
                    print(f"Unexpected filtered output format: {filtered_line}")
                    return None, None

                # Extract MMSE value
                if mmse_line.startswith("MMSE: "):
                    mmse = float(mmse_line.replace("MMSE: ", ""))
                else:
                    print(f"Unexpected MMSE format: {mmse_line}")
                    return None, None

                return filtered_output, mmse
            else:
                print(f"Error: output.txt doesn't contain enough lines: {len(lines)}")
                print(lines[0])
                print(lines[1])
                return None, None

    except Exception as e:
        print(f"Error extracting output: {e}")
        return None, None


noise_types = ["white", "pink", "brown", "none"]


# Function to run MARS and capture the output
def run_mars(test_num):
    noise_type = noise_types[test_num % len(noise_types)]
    print(f"Running test {test_num + 1} with {noise_type}...")

    # Generate desired and noisy signals for the current test
    generate_signals(noise_type=noise_type)

    # Run the MARS command using subprocess
    command = ["java", "-jar", mars_jar, asm_file]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the process was successful
    if result.returncode == 0:
        # Save the output to a file named output_<test_number>.txt
        output_filename = os.path.join(output_dir, f"output_{test_num}.txt")
        with open(output_filename, "w") as f:
            f.write(result.stdout)

        # Extract filtered output and MMSE for the next test
        filtered_output, mmse = extract_output()

        return filtered_output, mmse
    else:
        print(f"Test {test_num} failed.")
        return None, None


# Loop to run the tests sequentially and store the results
for i in range(num_tests):
    # Run the Mars command for each test

    filtered_output, mmse = run_mars(i + 1)

    # If successful, we can process the filtered output and MMSE
    if filtered_output is not None and mmse is not None:
        print(f"Test {i + 1} completed. MMSE: {mmse}")
        print(f"Filtered Output: {filtered_output}")
        all_mmse.append(mmse)
        all_filtered_outputs.append(filtered_output)

    else:
        print(f"Test {i + 1} failed.")

print(f"Finished {num_tests} tests. Results stored in '{output_dir}' directory.")


# Convert to numpy arrays for analysis
all_mmse = np.array(all_mmse)
all_filtered_outputs = np.array(
    all_filtered_outputs
)  # shape = (num_tests, signal_length)

# Compute summary statistics
print("\n=== SUMMARY ===")
print(f"Total tests run: {len(all_mmse)}")
print(f"Mean MMSE: {np.mean(all_mmse):.4f}")
print(f"Std MMSE: {np.std(all_mmse):.4f}")
print(f"Min MMSE: {np.min(all_mmse):.4f}")
print(f"Max MMSE: {np.max(all_mmse):.4f}")

# Compute mean filtered output across tests (optional)
mean_filtered = np.mean(all_filtered_outputs, axis=0)
print(f"\nMean filtered output across tests:\n{mean_filtered}")

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

# 1️⃣ MMSE across tests
plt.figure(figsize=(8, 4))
plt.plot(all_mmse, marker="o")
plt.title("MMSE Across Tests")
plt.xlabel("Test Number")
plt.ylabel("MMSE")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2️⃣ Histogram of MMSE distribution
plt.figure(figsize=(6, 4))
plt.hist(all_mmse, bins=10, edgecolor="black")
plt.title("MMSE Distribution")
plt.xlabel("MMSE Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 3️⃣ Mean filtered output vs. index
plt.figure(figsize=(8, 4))
plt.plot(mean_filtered, marker="s", color="orange")
plt.title("Mean Filtered Output Across Tests")
plt.xlabel("Sample Index")
plt.ylabel("Filtered Value")
plt.grid(True)
plt.tight_layout()
plt.show()
