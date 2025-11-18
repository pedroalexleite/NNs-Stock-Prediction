import subprocess

scripts = [
    "u-lstm.py",
    "u-gru.py",
    "m-lstm.py",
    "m-gru.py",
    "t-lstm.py",
    "t-gru.py"
]

output_file = "/Users/pedroalexleite/Desktop/NNs-Stock-Prediction/Data/model1.txt"

with open(output_file, "w") as f:
    for script in scripts:
        f.write(f"===== Running {script} =====\n")
        f.flush()
        print(f"===== Running {script} =====")
        try:
            process = subprocess.Popen(
                ["python3", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            for line in process.stdout:
                print(line, end="")
                f.write(line)
            process.wait()
            f.write("\n\n")
        except Exception as e:
            f.write(f"Error running {script}: {str(e)}\n")
            print(f"Error running {script}: {str(e)}")
