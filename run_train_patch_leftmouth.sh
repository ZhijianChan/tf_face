python train_model.py \
 --logs_base_dir   /home/chenzhijian/code/tf_face/logs \
 --models_base_dir /home/chenzhijian/code/tf_face/models  \
 --model_def       models.inception_resnet_v1_6c \
 --lfw_file_ext    _leftmouth_.jpg \
 --lfw_pairs       /exports_data/czj/data/lfw/files/pairs.txt  \
 --lfw_dir         /exports_data/czj/data/lfw/lfw_aligned/ \
 --lfw_batch_size  100 \
 --data_dir        /exports_data/czj/data/casia/casia_aligned/ \
 --imglist         /exports_data/czj/data/casia/files/train_set_leftmouth.txt  \
 --image_size      160 \
 --optimizer       RMSPROP \
 --lr              -1 \
 --lr_schedule_file lr_decay_4.txt \
 --batch_size      90 \
 --epoch_size      1000 \
 --max_num_epochs  80 \
 --keep_prob       0.8 \
 --weight_decay    5e-5 \
 --center_loss_factor 0 \
 --gpu_id 4
