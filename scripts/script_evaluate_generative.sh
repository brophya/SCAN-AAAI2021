for dset in zara1 zara2 hotel eth univ
do
python -u train_generative.py \
--test_only \
--dset_name univ \
--obs_len 8 \
--pred_len 12 \
--delim "\t" \
--model_type generative_spatial_temporal \
--domain_parameter 2 \
--batch_size 32 \
--noise_type uniform \
--noise_mix_type sample \
--num_epochs 200 \
--eval_batch_size 128 \
--best_k 5 \
--num_traj 20 \
--l 0.001 \
--delta_bearing 30 \
--delta_heading 30 \
--encoder_dim 32 \
--decoder_dim 32 \
--attention_dim 32 \
--embedding_dim 16 \
--d_spatial \
--d_type local 
done
