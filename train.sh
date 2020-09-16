python -u train.py \
--dset_name zara2 \
--delim tab \
--model_type generative_spatial_temporal \
--domain_type learnable \
--domain_init_type constant \
--best_k 1 \
--weight_sim 0 \
--delta_bearing 30 \
--delta_heading 30 \
--gpu_id 1 \
--log_results 
       


