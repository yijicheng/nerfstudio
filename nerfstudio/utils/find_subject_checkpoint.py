import os
import sys
from pathlib import Path

def _find_subject_ckpt(subject, output_dir, method_name, max_num_iterations, max_num_outer_epochs):

    if max_num_outer_epochs == 1:
        experiment_name = subject
    else:
        experiment_name = subject + f"_{max_num_outer_epochs}-{max_num_outer_epochs-1}"
    
    method_dir = Path(f"{output_dir}/{experiment_name}/{method_name}")
    try:
        timestamp_dirs = sorted(os.listdir(method_dir), key=lambda x: int(x.replace("-", "").replace("_", "")), reverse=True)
        for timestamp_dir in timestamp_dirs:
            checkpoint_dir = Path(timestamp_dir) / "nerfstudio_models"
            checkpoint_path = method_dir / checkpoint_dir / f"step-{max_num_iterations-1:09d}.ckpt"
            if checkpoint_path.exists():
                return checkpoint_path
            else:
                continue
        return None
    except:
        return None

if __name__ == '__main__':
    subject = sys.argv[1]
    output_dir = sys.argv[2]
    method_name = sys.argv[3]
    max_num_iterations = int(sys.argv[4])
    max_num_outer_epochs = int(sys.argv[5])

    print(_find_subject_ckpt(subject, output_dir, method_name, max_num_iterations, max_num_outer_epochs))