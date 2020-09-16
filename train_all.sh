for dset in zara2 univ zara1 hotel eth 
do 
    for k in 5 10 20
    do 
        for lambda in 0 0.0001 0.0005 0.001 
            do 
                echo "Training generative_spatial_temporal on dataset ${dset} with domain type learnable ${param} constant"
                python -u train.py \
                --dset_name ${dset} \
                --delim tab \
                --model_type generative_spatial_temporal \
                --domain_type learnable \
                --param_domain 5 \
                --domain_init_type constant \
                --best_k ${k} \
                --num_traj 20 \
                --weight_sim ${lambda} \
                --gpu_id 2 \
                --log_results > log/${dset}_generative_spatial_temporal_learnable_${param}_constant_${k}_${lambda}.out 
            done   
        
    done 
done 

                
    
