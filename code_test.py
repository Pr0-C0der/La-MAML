# take the loops from kaggle notebook. also enable cuda when running on VM. also change the data path and other variables accordingly
import subprocess
import sys
import os
# Function to install requirements
def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Install requirements
install_requirements()

SEED_LIST = [42, 43, 44]  # Example seed values
# Loop over the parameters
for SEED in SEED_LIST:
    # Construct the command
    command = [
        "python3", "main.py",
        "--data_path", "data/",
        "--log_every","100",
        "--dataset", "tinyimagenet",
        "--log_dir", "logs/",
        "--model", "lamaml_cifar",
        "--expt_name", "lamaml",
        "--memories", "400",
        "--batch_size", "10",
        "--replay_batch_size", "10",
        "--n_epochs", "1",
        "--opt_lr", "0.4",
        "--alpha_init", "0.1",
        "--opt_wt", "0.1",
        "--glances", "2",
        "--loader", "class_incremental_loader",
        "--increment", "5",
        "--arch", "pc_cnn",
        "--cifar_batches", "5",
        "--learn_lr",
        "--log_every", "3125",
        "--second_order",
        "--class_order", "random",
        "--seed", str(SEED),
        "--grad_clip_norm", "1.0",
        "--calc_test_accuracy",
        "--validation", "0.1"
    ]
    
    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Print the output for debugging purposes (optional)
    print(result.stdout)
    print(result.stderr)

# Keep variables as strings