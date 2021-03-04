for dset in zara1 zara2 univ eth hotel
do
python -u train_deterministic.py \
--dset_name $dset \
--test_only \
--obs_len 8 \
--pred_len 12 \
--delim "\t" \
--model_type spatial_temporal \
--domain_parameter 5 \
--batch_size 32 \
--eval_batch_size 32 \
--delta_bearing 30 \
--delta_heading 30 \
--encoder_dim 32 \
--decoder_dim 32 \
--attention_dim 32 \
--embedding_dim 16 \
--use_scene_context 
done
