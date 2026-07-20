# src/progress.py — progress bar ASCII ke arah MAX_ROUNDS dan TARGET_LOSS
from src import config


def make_progress_bar(fraction: float, width: int = 24) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    bar = "█" * filled + "-" * (width - filled)
    return f"[{bar}] {fraction * 100:5.1f}%"


def print_progress(round_num: int, eval_loss: float):
    round_fraction = round_num / config.MAX_ROUNDS

    if config.BASELINE_LOSS > config.TARGET_LOSS:
        loss_fraction = (config.BASELINE_LOSS - eval_loss) / (config.BASELINE_LOSS - config.TARGET_LOSS)
    else:
        loss_fraction = 1.0 if eval_loss <= config.TARGET_LOSS else 0.0

    print(f"[server] Ronde   {round_num:3d}/{config.MAX_ROUNDS}  {make_progress_bar(round_fraction)}")
    print(f"[server] Loss    {eval_loss:.4f} -> target {config.TARGET_LOSS}  {make_progress_bar(loss_fraction)}")