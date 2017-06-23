# pretrained face model
#python save_weights.py \
#    --meta_file ../models/20170602-102625/model-20170602-102625.meta \
#    --ckpt_file ../models/20170602-102625/model-20170602-102625.ckpt-20000 \
#    --save_path pretrained/pretrained_face.npy

# pretrained nose model
python save_weights.py \
    --meta_file ../models/20170620-121822/model-20170620-121822.meta \
    --ckpt_file ../models/20170620-121822/model-20170620-121822.ckpt-53000 \
    --save_path pretrained/pretrained_nose.npy

# pretrained lefteye model
#python save_weights.py \
#    --meta_file ../models/20170620-121734/model-20170620-121734.meta \
#    --ckpt_file ../models/20170620-121734/model-20170620-121734.ckpt-54000 \
#    --save_path pretrained/pretrained_lefteye.npy

# pretrained rightmouth model
#python save_weights.py \
#    --meta_file ../models/20170620-121936/model-20170620-121936.meta \
#    --ckpt_file ../models/20170620-121936/model-20170620-121936.ckpt-54000 \
#    --save_path pretrained/pretrained_rightmouth.npy
