python -u train_deterministic.py \
--dset_name zara1 \
--obs_len 8 \
--pred_len 12 \
--delim tab \
--model_type spatial \
--domain_type learnable \
--domain_parameter 5 \
--batch_size 32 \
--lr 0.001 \
--num_epochs 100 \
--delta_bearing 30 \
--delta_heading 30 \
--gpu_id 3 \
--use_scene_context #> log/zara1_8_12_2-$val-log.out &
       


