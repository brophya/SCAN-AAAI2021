for dset in zara1 zara2 eth hotel univ
do
python -u train_deterministic.py \
--dset_name $dset \
--obs_len 8 \
--pred_len 12 \
--delim "\t" \
--model_type spatial_temporal \
--scheduler \
--domain_parameter 2 \
--batch_size 32 \
--eval_batch_size 128 \
--num_epochs 200 \
--delta_bearing 30 \
--delta_heading 30 \
--encoder_dim 32 \
--decoder_dim 32 \
--attention_dim 32 \
--embedding_dim 16 \
--lr 1e-03 \
--gpu_id 3 \
--use_scene_context 
done
