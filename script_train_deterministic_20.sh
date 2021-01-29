for dset in zara1 zara2 eth hotel univ
do
nohup python -u train_deterministic.py \
--dset_name $dset \
--obs_len 8 \
--pred_len 8 \
--delim tab \
--model_type spatial_temporal \
--domain_type learnable \
--domain_parameter 2 \
--batch_size 32 \
--lr 0.001 \
--num_epochs 100 \
--delta_bearing 30 \
--delta_heading 30 \
--gpu_id 3 \
--use_scene_context > log/$dset-8_8_2_30_log.out &
echo "Pausing execution.."
sleep 2m
echo "Resuming.."
done 
       


