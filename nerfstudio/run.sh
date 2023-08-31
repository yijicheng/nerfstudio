# python scripts/train.py \
#     kplanes-ngp \
#     --machine.num-devices=8 \
#     --vis=tensorboard \
#     --output-dir=../outputs \
#     --pipeline.datamanager.train-num-rays-per-batch=8192 \
#     --pipeline.datamanager.eval-num-rays-per-batch=8192 \
#     --pipeline.datamanager.images-on-gpu=True \
#     --optimizers.field.color_net.optimizer.max_norm=0.1 \
#     --optimizers.field.sigma_net.optimizer.max_norm=0.1 \
#     --mixed-precision=False \
#     --logging.local-writer.max-log-size=2 \
#     --log-gradients=True \
#     blender-data \
#     --data ../../data/blender/lego

python scripts/train_parallel_subject.py \
    kplanes-ngp-multiple-fitting \
    --machine.num-devices=8 \
    --vis=tensorboard \
    --output-dir=../outputs/new_codebase \
    --pipeline.datamanager.train-num-rays-per-batch=8192 \
    --pipeline.datamanager.eval-num-rays-per-batch=8192 \
    --pipeline.datamanager.images-on-gpu=True \
    --max-num-iterations=5001 \
    --max-num-outer-epochs=30 \
    --steps-per-eval-image=1000 \
    --steps-per-eval-batch=1000 \
    --steps-per-eval-all-images=5000 \
    --steps-per-save=5000 \
    --optimizers.field.color_net.optimizer.max_norm=0.1 \
    --optimizers.field.sigma_net.optimizer.max_norm=0.1 \
    --mixed-precision=False \
    --logging.local-writer.max-log-size=1 \
    --log-gradients=True \
    --num_train_subjects=8 \
    rodin
