GPUId=1
CUDA_VISIBLE_DEVICES=$GPUId
python train_model.py \
 --model_def            models.inception_resnet_v1 \
 --lfw_file_ext         _face_.jpg \
 --lfw_pairs            /exports_data/czj/data/lfw/files/pairs.txt  \
 --lfw_dir              /exports_data/czj/data/lfw/lfw_aligned/ \
 --lfw_batch_size       100 \
 --data_dir             /exports_data/czj/data/ms_celeb_1m/ms_celeb_1m_aligned/ \
 --imglist_path         /exports_data/czj/data/ms_celeb_1m/files/train_set.txt \
 --image_size           160 \
 --optimizer            ADAM \
 --lr                   -1 \
 --lr_schedule_file     lr_decay.txt \
 --batch_size           128 \
 --max_num_epochs       7 \
 --keep_prob            0.8 \
 --weight_decay         5e-5 \
 --center_loss_factor   0 \
 --center_loss_alpha    0.9 \
 --gpu_id               $GPUId \
 --random_flip \
 --random_crop \
 --logs_base_dir        logs/models/inception_resnet_v1/baseline
