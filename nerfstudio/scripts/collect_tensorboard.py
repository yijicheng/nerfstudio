from pathlib import Path
import sys
import os
import shutil

def _find_subject_ckpt(subject, output_dir, method_name, max_num_iterations, max_num_outer_epochs, epoch):

    experiment_name = subject + f"_{max_num_outer_epochs}-{epoch}"
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

with open("/mnt/blob/data/rodin/data_10persons.txt", 'r') as f:
    SEEN_SUBJECT_10 = f.read().splitlines()

with open("/mnt/blob/data/rodin/data_15persons.txt", 'r') as f:
    SEEN_SUBJECT_15 = f.read().splitlines()

with open("/mnt/blob/data/rodin/person_test_10.txt", 'r') as f:
    UNSEEN_SUBJECT = f.read().splitlines()

if __name__ == '__main__':

    # data
    data_name = sys.argv[1]
    if data_name in ["SEEN_SUBJECT_10", "SEEN_SUBJECT_15", "UNSEEN_SUBJECT"]:
        data = eval(data_name)
    else:
        data = [data_name]
    output_dir = "/mnt/blob/ckpt/stage1_parallel_subject_fitting_30ep_5000it_10ddp_fp32_scale_up_subject"

    tb_collect_dir = Path(sys.argv[2])
    tb_collect_dir = tb_collect_dir / output_dir.split('/')[-1]
    
    tb_out_collection = []
    for subject in data:
        
        tb_save_path = tb_collect_dir / subject
        tb_save_path.mkdir(parents=True, exist_ok=True)
        for epoch in range(30):
            subject_ckpt = _find_subject_ckpt(subject, output_dir, "multiple-fitting", 5001, 30, epoch)
            if subject_ckpt is not None:
                tb_out = list(subject_ckpt.parent.parent.rglob("events.out.tfevents.*"))[0]
                shutil.copy(tb_out, tb_save_path)
                print(f"from {tb_out} to {tb_save_path}")

    print(tb_collect_dir)
