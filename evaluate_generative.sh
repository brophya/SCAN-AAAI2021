python -u evaluate_generative.py \
--dset_name zara1 \
--delim tab \
--model_type generative_spatial_temporal \
--domain_type learnable \
--domain_parameter 5 \
--eval_batch_size 1 \
--best_k 20 \
--num_traj 20 \
--weight_sim 0 \
--lr 0.001 \
--delta_bearing 30 \
--delta_heading 30 \
--gpu_id 3 \
--use_scene_context \
       

