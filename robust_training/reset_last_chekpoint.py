import os
import re
import shutil

def update_checkpoints(path: str, num: int):
    """
    Deletes the last.pth.tar and the last `num` checkpoint files,
    then copies the latest remaining checkpoint as last.pth.tar.
    """
    # Define regex pattern to match checkpoint files
    pattern = re.compile(r"checkpoint-(\d+)\.pth\.tar$")
    
    # Collect all checkpoint files and their numbers
    checkpoints = []
    for f in os.listdir(path):
        match = pattern.match(f)
        if match:
            checkpoints.append((int(match.group(1)), f))
    
    if not checkpoints:
        print("No checkpoint files found.")
        return
    
    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: x[0])
    
    # Delete last.pth.tar if it exists
    last_path = os.path.join(path, "last.pth.tar")
    if os.path.exists(last_path):
        os.remove(last_path)
        print(f"Deleted: {last_path}")
    
    # Delete the last `num` checkpoint files
    to_delete = checkpoints[-num:]
    for _, f in to_delete:
        os.remove(os.path.join(path, f))
        print(f"Deleted: {f}")
    
    # Determine new latest checkpoint
    remaining = checkpoints[:-num]
    if not remaining:
        print("No remaining checkpoints to copy as last.pth.tar.")
        return
    
    latest_checkpoint = remaining[-1][1]
    src = os.path.join(path, latest_checkpoint)
    dst = os.path.join(path, "last.pth.tar")
    
    # Copy latest checkpoint to last.pth.tar
    shutil.copy2(src, dst)
    print(f"Copied {latest_checkpoint} → last.pth.tar")

paths = [
    # "/home/ashtomer/projects/ares/results/models/convnext_small/convnext_small_eps-0_None_seed-None",
    # "/home/ashtomer/projects/ares/results/models/convnext_small/convnext_small_eps-1_l2_seed-1",
    # "/home/ashtomer/projects/ares/results/models/convnext_small/convnext_small_eps-1_l2_seed-2",
    "/home/ashtomer/projects/ares/results/models/convnext_small/convnext_small_eps-1_linf_seed-1"
]
# Example usage:
for path in paths:
    update_checkpoints(path, 3)
    print(f"Updated checkpoints in: {path}")
    
    