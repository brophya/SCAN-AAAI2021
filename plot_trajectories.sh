for k in 1 5 10 20
do 
    for lambda in 0 0.0001 0.0005 0.001 0.01 
    do 
        python -u plot_trajectories.py \
        --plot_densities \
        --dset_name=zara1 \
        --delim=tab \
        --model_type=generative_spatial_temporal \
        --best_k=${k} \
        --num_traj=300 \
        --gpu_id=2 \
        --weight_sim=${lambda} \
        --param_domain=2 
    done 
done 
