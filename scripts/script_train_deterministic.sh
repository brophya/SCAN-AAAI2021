for dset in eth
do
for model in spatial_temporal
do
nohup \
python -u train_deterministic.py \
--dset_name $dset \
--obs_len 8 \
--pred_len 12 \
--delim "\t" \
--model_type $model \
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
--scheduler > log.$dset.out &
done
done
