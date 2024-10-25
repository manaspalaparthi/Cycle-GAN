import torch
import config

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


from torch.utils.tensorboard import SummaryWriter
import os

def get_next_experiment_number(log_dir="runs"):
    # Create the directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Get a list of existing experiment folders
    existing_experiments = [d for d in os.listdir(log_dir) if d.startswith("experiment")]
    # Extract the experiment numbers and find the next number
    experiment_numbers = [int(d.split("experiment")[1]) for d in existing_experiments if d.split("experiment")[1].isdigit()]
    next_experiment_number = max(experiment_numbers) + 1 if experiment_numbers else 1
    return next_experiment_number