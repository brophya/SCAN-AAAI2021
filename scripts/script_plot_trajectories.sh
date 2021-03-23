for k in 1 5 10 20
do
for l in 0 0.001 0.1
do
python -u plot_trajectories.py \
--dset_name zara1 \
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
--best_k $k  \
--l2_loss_weight 0.5 \
--num_traj 10 \
--l $l \
--lr_g 1e-03 \
--lr_d 1e-03 \
--scheduler \
--delta_bearing 30 \
--delta_heading 30 \
--encoder_dim 32 \
--decoder_dim 32 \
--attention_dim 32 \
--embedding_dim 16 \
--d_type local \
--d_spatial 
done
done
