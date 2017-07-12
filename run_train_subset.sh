python train_model_subset.py \
 --model_def            models.inception_resnet_v1 \
 --lfw_file_ext         _face_.jpg \
 --lfw_pairs            /exports_data/czj/data/lfw/files/pairs.txt  \
 --lfw_dir              /exports_data/czj/data/lfw/lfw_aligned/ \
 --lfw_batch_size       100 \
 --data_dir             /exports_data/czj/data/ms_celeb_1m/ms_celeb_1m_aligned/ \
 --imglist              /exports_data/czj/data/ms_celeb_1m/files/train_set.txt \
 --image_size           160 \
 --optimizer            ADAM \
 --lr                   -1 \
 --lr_schedule_file     lr_decay_subset.txt \
 --batch_size           90 \
 --epoch_size           1000 \
 --max_num_epochs       80 \
 --keep_prob            0.8 \
 --weight_decay         5e-5 \
 --center_loss_factor   0 \
 --center_loss_alpha    0.9 \
 --gpu_id               3 \
 --random_flip \
 --random_crop \
 --logs_base_dir        logs/subset/