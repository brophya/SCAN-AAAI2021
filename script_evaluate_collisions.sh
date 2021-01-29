for dset in zara1 zara2 eth univ hotel
do
python -u evaluate_collisions.py \
--dset_name $dset \
--delim tab \
--model_type spatial_temporal \
--domain_type learnable \
--pred_len 12 \
--domain_parameter 5 \
--batch_size 32 \
--lr 0.001 \
--num_epochs 200 \
--delta_bearing 30 \
--delta_heading 30 
done
