for dset in univ
do
for p in 12
do
python -u train_deterministic.py \
--dset_name $dset \
--test_only \
--obs_len 8 \
--pred_len $p \
--delim "\t" \
--model_type spatial \
--domain_parameter 2 \
--batch_size 32 \
--eval_batch_size 1 \
--delta_bearing 30 \
--delta_heading 30 \
--encoder_dim 32 \
--decoder_dim 32 \
--attention_dim 32 \
--embedding_dim 16 
done
done
