dataset=2
if [ $dataset -eq 1 ]; then
    data_root=/exports_data/czj/data/facescrub/facescrub_aligned/
    templates_file=../files/facescrub_selected_features_list.json
else
    data_root=/exports_data/czj/data/megaface/megaface_aligned/FlickrFinal2/
    templates_file=../files/megaface_features_list.json_1000000_1
fi

CUDA_VISIBLE_DEVICES=4 
python extract_feature.py \
    $data_root \
    $templates_file \
    pretrained/weights_c.npy \
    models.inception_resnet_v1 \
    _inception_v1.bin \
    $CUDA_VISIBLE_DEVICES
