from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()

# get the path to the TensorBoard log directory
log_dir = Path(input("enter path: "))
event_file = [f for f in os.listdir(log_dir) if f.startswith("events.out")][0]
event_path = log_dir / event_file

# load the TensorBoard event file
event_acc = EventAccumulator(event_path)
event_acc.Reload()

# scalar tags
print("Logged scalars:", event_acc.Tags()["scalars"])

def save_fig(tag, event_acc, save_dir):
    events = event_acc.Scalars(tag)

    # convert to lists for easier handling
    steps = [e.step for e in events]
    values = [e.value for e in events]

    # visualize
    plt.figure(figsize=(10, 5))
    plt.plot(steps, values, label=tag)
    plt.xlabel("Training Step")
    plt.ylabel(tag)
    plt.title(f"{tag} over Training Steps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_dir, dpi=300)

graph_dir = PROJECT_ROOT / "graphs" / "DPO_pairwise" / "TinyLlama" / "TinyLlama-1.1B-Chat-v1.0"
os.makedirs(graph_dir, exist_ok=True)

for tag in event_acc.Tags()["scalars"]:
    print(f"Processing tag: {tag}")
    save_dir = graph_dir / f"{tag.replace('/', '_')}.png"
    save_fig(tag, event_acc, str(save_dir))


plt.figure(figsize=(10, 5))

events = event_acc.Scalars("train/rewards/chosen")
steps = [e.step for e in events]
values = [e.value for e in events]
plt.plot(steps, values, label="train/rewards/chosen")
events = event_acc.Scalars("train/rewards/rejected")
values = [e.value for e in events]
plt.plot(steps, values, label="train/rewards/rejected")
plt.xlabel("Training Step")
plt.ylabel("Rewards")
plt.title("Rewards over Training Steps")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(str(graph_dir / "train_rewards.png"), dpi=300)
