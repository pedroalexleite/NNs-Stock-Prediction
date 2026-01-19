import subprocess

scripts = [
    "U-LSTM.py",
    "U-GRU.py",
    "M-LSTM.py",
    "M-GRU.py",
    "T-LSTM.py",
    "T-GRU.py"
]

output_file = "/Users/pedroalexleite/Desktop/Tese/Dados/Model1.csv"

with open(output_file, "w") as f:
    for script in scripts:
        f.write(f"===== Running {script} =====\n")
        try:
            result = subprocess.run(
                ["python3", script],
                capture_output=True,
                text=True
            )
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- ERRORS ---\n")
                f.write(result.stderr)
            f.write("\n\n")
        except Exception as e:
            f.write(f"Error running {script}: {str(e)}\n")
