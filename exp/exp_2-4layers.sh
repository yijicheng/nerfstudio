
cd nerfstudio

# stage1 config
# subject_file=/mnt/blob/data/rodin/data/data_10persons.txt
subject_file=/mnt/blob2/avatar/person_list/2.18_hd.txt
num_train_subjects=8
stage1_output_dir=../outputs/stage1_parallel_subject_fitting_30ep_5000it_10ddp_fp32_2-4layers_2e-4_2e-2_lr
stage1_method_name=kplanes-ngp-multiple-fitting
stage1_max_num_iterations=5001
stage1_max_num_outer_epochs=30
stage1_steps_per_eval_image=1000
stage1_steps_per_eval_batch=1000
stage1_steps_per_eval_all_images=5000
stage1_steps_per_save=5000
grids_lr=2e-2
decoders_lr=2e-4
sigma_grad_clip=0.1
color_grad_clip=0.3

# stage2 config
unseen_subject_file=/mnt/blob/data/rodin/data/person_test_10.txt
stage2_method_name=kplanes-ngp
stage2_max_num_iterations=150001
stage2_steps_per_eval_image=5000
stage2_steps_per_eval_batch=5000
stage2_steps_per_eval_all_images=30000
stage2_steps_per_save=30000

# 获取整个字符串  
str_pairs=$1
# e.g. str_pairs=stage1_train=seen;stage2_train=unseen
  
# 使用分号分隔字符串对  
IFS=';' read -ra pairs <<< "$str_pairs"

# 遍历每个字符串对  
for pair in "${pairs[@]}"
do  
    # 使用等号分隔键和值  
    IFS='=' read -ra parts <<< "$pair"  
    task="${parts[0]}"  
    data="${parts[1]}"

    # 打印键和值  
    echo "task: $task, data: $data"

    # stage1
    if [ "$task" = "stage1_train" ] && [ "$data" = "seen" ]
    then
        python scripts/train_parallel_subject.py \
            ${stage1_method_name} \
            --machine.num-devices=8 \
            --vis=tensorboard \
            --output-dir=${stage1_output_dir} \
            --pipeline.datamanager.train-num-rays-per-batch=8192 \
            --pipeline.datamanager.eval-num-rays-per-batch=8192 \
            --pipeline.datamanager.images-on-gpu=True \
            --max-num-iterations=${stage1_max_num_iterations} \
            --max-num-outer-epochs=${stage1_max_num_outer_epochs} \
            --steps-per-eval-image=${stage1_steps_per_eval_image} \
            --steps-per-eval-batch=${stage1_steps_per_eval_batch} \
            --steps-per-eval-all-images=${stage1_steps_per_eval_all_images} \
            --steps-per-save=${stage1_steps_per_save} \
            --optimizers.field.grids.optimizer.lr=${grids_lr} \
            --optimizers.field.sigma-net.optimizer.lr=${decoders_lr} \
            --optimizers.field.color-net.optimizer.lr=${decoders_lr} \
            --optimizers.field.sigma_net.optimizer.max_norm=${sigma_grad_clip} \
            --optimizers.field.color_net.optimizer.max_norm=${color_grad_clip} \
            --mixed-precision=False \
            --logging.local-writer.max-log-size=1 \
            --log-gradients=True \
            --num_train_subjects=${num_train_subjects} \
            --pipeline.model.sigma-decoder-layers=2 \
            --pipeline.model.color-decoder-layers=4 \
            rodin
        wait
    fi

    # stage2 unseen_subject 
    if [ "$task" = "stage2_train" ] && [ "$data" = "unseen" ]
    then
        count=0
        while read subject && [ ${count} -le 9]  
        do  
            echo "Line ${count}: ${subject}"

            decoder_ckpt=`python utils/find_subject_checkpoint.py ${subject} ${stage1_output_dir} ${stage1_method_name} ${stage1_max_num_iterations} ${stage1_max_num_outer_epochs}`
            stage2_output_dir="${stage1_output_dir}_stage2"

            CUDA_VISIBLE_DEVICES=${count} python scripts/train.py \
                kplanes-ngp \
                --machine.num-devices=1 \
                --vis=tensorboard \
                --output-dir=${stage2_output_dir} \
                --experiment_name=${subject} \
                --pipeline.datamanager.train-num-rays-per-batch=8192 \
                --pipeline.datamanager.eval-num-rays-per-batch=8192 \
                --pipeline.datamanager.images-on-gpu=True \
                --max-num-iterations=${stage2_max_num_iterations} \
                --steps-per-eval-image=${stage2_steps_per_eval_image} \
                --steps-per-eval-batch=${stage2_steps_per_eval_batch} \
                --steps-per-eval-all-images=${stage2_steps_per_eval_all_images} \
                --steps-per-save=${stage2_steps_per_save} \
                --optimizers.field.grids.optimizer.lr=${grids_lr} \
                --pipeline.model.decoder-checkpoint=${decoder_ckpt} \
                --pipeline.model.freeze-decoder=True \
                --mixed-precision=False \
                --logging.local-writer.max-log-size=1 \
                --log-gradients=True \
                --pipeline.model.sigma-decoder-layers=2 \
                --pipeline.model.color-decoder-layers=4 \
                rodin \
                data=/mnt/blob2/render_output_hd/${subject} &


            ((count++))  
        done < <(head -n ${num_train_subjects} ${unseen_subject_file}) 
        wait
    fi

    # eval stage2 unseen_subject 
    if [ "$task" = "stage2_eval" ] && [ "$data" = "unseen" ]
    then
        count=0
        while read subject && [ ${count} -le 9 ]  
        do  
            echo "Line ${count}: ${subject}"
            stage2_ckpt=`python utils/find_subject_checkpoint.py ${subject} ${stage2_output_dir} ${stage2_method_name} ${stage2_max_num_iterations} 1`
            config_path=dirname "$(dirname "$(realpath ${stage2_ckpt})")"
            config_path="${config_path}/config.yml"
            eval_stage2_output_dir=stage2_output_dir="${stage2_output_dir}_eval"
            CUDA_VISIBLE_DEVICES=${count} ns-eval \
                --load-config=${config_path} \
                --output-path=${eval_output_dir}/${subject}/output.json \
                --render-output-path=${eval_output_dir}/${subject} &

            ((count++))  
        done < <(head -n ${num_train_subjects} ${unseen_subject_file})
        wait
    fi


    # stage2 seen_subject
    if [ "$task" = "stage2_train" ] && [ "$data" = "seen" ]
    then
        count=0
        while read subject && [ ${count} -le $((num_train_subjects - 1)) ]  
        do  
            echo "Line ${count}: ${subject}"

            decoder_ckpt=`python utils/find_subject_checkpoint.py ${subject} ${stage1_output_dir} ${stage1_method_name} ${stage1_max_num_iterations} ${satge1_max_num_outer_epochs}`
            stage2_output_dir="${stage1_output_dir}_stage2"

            python scripts/train.py \
                ${stage2_method_name} \
                --machine.num-devices=1 \
                --vis=tensorboard \
                --output-dir=${stage2_output_dir} \
                --experiment_name=${subject} \
                --pipeline.datamanager.train-num-rays-per-batch=8192 \
                --pipeline.datamanager.eval-num-rays-per-batch=8192 \
                --pipeline.datamanager.images-on-gpu=True \
                --max-num-iterations=${stage2_max_num_iterations} \
                --steps-per-eval-image=${stage2_steps_per_eval_image} \
                --steps-per-eval-batch=${stage2_steps_per_eval_batch} \
                --steps-per-eval-all-images=${stage2_steps_per_eval_all_images} \
                --steps-per-save=${stage2_steps_per_save} \
                --optimizers.field.grids.optimizer.lr=${grids_lr} \
                --pipeline.model.decoder-checkpoint=${decoder_ckpt} \
                --pipeline.model.freeze-decoder=True \
                --mixed-precision=False \
                --logging.local-writer.max-log-size=1 \
                --log-gradients=True \
                --pipeline.model.sigma-decoder-layers=2 \
                --pipeline.model.color-decoder-layers=4 \
                rodin \
                data=/mnt/blob2/render_output_hd/${subject} &

            ((count++))  
        done < <(head -n ${num_train_subjects} ${subject_file}) 
        wait
    fi

    # eval stage2 seen_subject 
    if [ "$task" = "stage2_eval" ] && [ "$data" = "seen" ]
    then
        count=0
        while read subject && [ ${count} -le $((num_train_subjects - 1)) ]  
        do  
            echo "Line ${count}: ${subject}"
            stage2_ckpt=`python utils/find_subject_checkpoint.py ${subject} ${stage2_output_dir} ${stage2_method_name} ${stage2_max_num_iterations} 1`
            config_path=dirname "$(dirname "$(realpath ${stage2_ckpt})")"
            config_path="${config_path}/config.yml"
            eval_stage2_output_dir=stage2_output_dir="${stage2_output_dir}_eval"
            ns-eval \
                --load-config=${config_path} \
                --output-path=${eval_output_dir}/${subject}/output.json \
                --render-output-path=${eval_output_dir}/${subject} &

            ((count++))  
        done < <(head -n ${num_train_subjects} ${subject_file}) 
        wait
    fi

done  
