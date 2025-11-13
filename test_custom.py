import os
import shutil
import subprocess
import numpy as np

# ==== CONFIG ====
mars_jar = "Mars4_5.jar"
asm_file = "main.asm"
tests_root = "tests_custom"
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)

# ==== FUNCTIONS ====


def run_mars():
    """Run MARS simulator and return filtered output + MMSE parsed from output.txt.
    Handles normal results and error outputs (like 'Error: size not match').
    """
    command = ["java", "-jar", mars_jar, asm_file]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ MARS execution failed.")
        print(result.stderr)
        return None, None

    try:
        with open("output.txt", "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        # --- handle assembly error output case ---
        for line in lines:
            if line.startswith("Error:"):
                print(f"⚠️ MARS reported error: {line}")
                # Return a special marker tuple (error_message, None)
                return line, None

        filtered_output, mmse = None, None

        for line in lines:
            if line.startswith("Filtered output: "):
                filtered_output = list(
                    map(float, line.replace("Filtered output: ", "").split())
                )
            elif line.startswith("MMSE: "):
                mmse = float(line.replace("MMSE: ", ""))

        if filtered_output is None or mmse is None:
            raise ValueError("Output file missing required data.")

        return filtered_output, mmse

    except FileNotFoundError:
        print("⚠️ output.txt not found (MARS may not have written output).")
        return None, None
    except Exception as e:
        print(f"⚠️ Error parsing output.txt: {e}")
        return None, None


def run_test_case(test_dir):
    """Run one test case and print detailed comparison results."""
    desired_path = os.path.join(test_dir, "desired.txt")
    input_path = os.path.join(test_dir, "input.txt")
    expected_path = os.path.join(test_dir, "expected.txt")

    print(f"\n=== 🧪 Running {test_dir} ===")

    # Copy input/desired into current working directory
    shutil.copy(desired_path, "desired.txt")
    shutil.copy(input_path, "input.txt")

    # Read inputs for display
    with open(desired_path) as f:
        desired = f.read().strip()
    with open(input_path) as f:
        noisy = f.read().strip()

    # Run MARS and capture its output
    filtered_output, mmse = run_mars()

    if filtered_output is None:
        print("❌ Test failed: could not get output from MARS.")
        return

    # Read expected results (optional)
    expected_filtered, expected_mmse = None, None
    if os.path.exists(expected_path):
        with open(expected_path) as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        for line in lines:
            if line.startswith("Filtered output: "):
                expected_filtered = list(
                    map(float, line.replace("Filtered output: ", "").split())
                )
            elif line.startswith("MMSE: "):
                expected_mmse = float(line.replace("MMSE: ", ""))

    # Display all
    print(f"📈 Desired:         {desired}")
    print(f"🔊 Input:           {noisy}")
    print(f"🏁 Output (y_hat):  {filtered_output}")
    print(f"📉 MMSE:            {mmse}")

    if expected_filtered is not None:
        print(f"✅ Expected y_hat:  {expected_filtered}")
        print(f"✅ Expected MMSE:   {expected_mmse}")

        # Compare numerically
        out_diff = np.allclose(filtered_output, expected_filtered, atol=1e-3)
        mmse_diff = np.isclose(mmse, expected_mmse, atol=1e-3)

        print(f"🔍 Match (Output):  {'✔️' if out_diff else '❌'}")
        print(f"🔍 Match (MMSE):    {'✔️' if mmse_diff else '❌'}")

    # Save raw output
    test_name = os.path.basename(test_dir)
    out_file = os.path.join(output_dir, f"{test_name}_output.txt")
    with open(out_file, "w") as f:
        f.write(f"Filtered output: {' '.join(map(str, filtered_output))}\n")
        f.write(f"MMSE: {mmse}\n")

    print(f"📁 Saved result → {out_file}")


# ==== MAIN LOOP ====

if __name__ == "__main__":
    test_folders = sorted(
        [
            os.path.join(tests_root, d)
            for d in os.listdir(tests_root)
            if os.path.isdir(os.path.join(tests_root, d))
        ]
    )

    print(f"Found {len(test_folders)} test cases in '{tests_root}'.")

    for test_dir in test_folders:
        run_test_case(test_dir)

    print("\n✅ All tests completed.")
