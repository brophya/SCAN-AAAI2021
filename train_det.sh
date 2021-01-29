python -u train_deterministic.py \
--dset_name zara1 \
--obs_len 8 \
--pred_len 12 \
--delim tab \
--model_type spatial_temporal \
--domain_type learnable \
--domain_parameter 5 \
--batch_size 16 \
--lr 0.001 \
--num_epochs 100 \
--delta_bearing 30 \
--delta_heading 30 \
--gpu_id 2 \
--use_scene_context 
       


