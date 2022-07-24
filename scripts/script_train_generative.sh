for k in 5
do
for lambda in 0
do
for dset in univ
do
#sleep 45s
#nohup \
python -u \
train_generative.py \
--dset_name univ \
--obs_len 8 \
--pred_len 12 \
--delim "\t" \
--model_type generative_spatial_temporal \
--domain_parameter 2 \
--batch_size 32 \
--noise_type gaussian \
--noise_mix_type ped \
--noise_dim 16 \
--num_epochs 200 \
--best_k $k \
--l2_loss_weight 0.5 \
--num_traj 20 \
--l $lambda \
--lr_g 1e-03 \
--lr_d 1e-03 \
--scheduler \
--delta_bearing 30 \
--delta_heading 30 \
--encoder_dim 32 \
--decoder_dim 32 \
--attention_dim 32 \
--embedding_dim 16 \
--eval_batch_size 128 \
--d_type local \
--d_spatial \
--d_domain
done
done
done

