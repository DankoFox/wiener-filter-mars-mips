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
def generate_signals():
    # Generate a random desired signal (e.g., 10 values)
    desired_signal = np.random.uniform(-1000, 1000, 10)
    desired_signal = np.round(desired_signal, 1)  # Round to 1 decimal place

    # Add some random white noise to the desired signal
    noise = np.random.normal(0, 0.5, desired_signal.shape)  # Mean = 0, stddev = 0.5
    noisy_signal = desired_signal + noise
    noisy_signal = np.round(noisy_signal, 1)  # Round noisy signal to 1 decimal place

    # Write desired signal and noisy signal to files
    with open("desired.txt", "w") as f:
        f.write(", ".join(map(str, desired_signal)) + "\n")

    with open("input.txt", "w") as f:
        f.write(", ".join(map(str, noisy_signal)) + "\n")

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


# Function to run MARS and capture the output
def run_mars(test_num):
    # Generate desired and noisy signals for the current test
    generate_signals()

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
    print(f"Running test {i + 1} of {num_tests}...")

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
