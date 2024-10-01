# Train a ViT-Small global retrieval model, initialized with DeiT-Small weights from ImageNet-1K
python3 -u train_deit.py  --backbone deit --aggregation gem --mining partial  --save_dir global_retrieval --lr 0.00001 --fc_output_dim 256 --train_batch_size 3 --infer_batch_size 96 --num_workers 8 --epochs_num 100 --patience 10 --negs_num_per_query 2 --queries_per_epoch 5000 --cache_refresh_rate 1000

# Use backbone="resnet50conv5" to train a resnet model with VG benchmark