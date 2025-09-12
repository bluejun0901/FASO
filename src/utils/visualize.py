from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()  # type: ignore

# get the path to the TensorBoard log directory
log_dir = PROJECT_ROOT / input("enter path: ")
event_acc = EventAccumulator(str(log_dir))
event_acc.Reload()

# scalar tags
print("Logged scalars:", event_acc.Tags()["scalars"])

def ema(values, alpha=0.98):
    """Exponential Moving Average smoothing"""
    smoothed = []
    m = None
    for v in values:
        m = v if m is None else alpha * m + (1 - alpha) * v
        smoothed.append(m)
    return smoothed

def save_fig(tag, event_acc, save_path):
    try:
        events = event_acc.Scalars(tag)
    except KeyError:
        return  # 태그가 없으면 스킵

    if not events:
        return

    steps = [e.step for e in events]
    values = [e.value for e in events]

    # (선택) step 기준 정렬
    if steps != sorted(steps):
        pairs = sorted(zip(steps, values))
        steps, values = zip(*pairs)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, values, label=f"{tag} (raw)")

    smoothed = ema(values, alpha=0.98)
    plt.plot(steps, smoothed, label=f"{tag} (EMA)")
    plt.xlabel("Training Step")
    plt.ylabel("Value")            # <- y라벨은 값 이름으로 통일
    plt.title(f"{tag} over Training Steps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()                    # <- 반드시 닫아주기

graph_dir = PROJECT_ROOT / input("input path: ")
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