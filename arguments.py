import argparse

parser = argparse.ArgumentParser()

def parse_arguments():
    # Dataset Parameters
    parser.add_argument('--dset_name', type=str, choices=['zara1','zara2','eth','hotel','univ','argoverse'], help="dataset")
    parser.add_argument('--obs_len', type=int, default=8, help="observation time window length")
    parser.add_argument('--pred_len', type=int, default=12, help="prediction time window length")
    parser.add_argument('--delim', type=str, default=" ", help="delimiter for parsing dataset")
    parser.add_argument('--drop_frames', default=0, type=int, help="track fragmentation drop frames")
    parser.add_argument('--augment_data', action='store_true', help="augment data by rotating frames randomly")

    # Model Parameters
    parser.add_argument('--model_type', type=str, default="spatial_temporal",choices=['vanilla', 'spatial', 'spatial_temporal', 'generative_vanilla', \
								'generative_spatial', 'generative_spatial_temporal'])
    parser.add_argument('--encoder_dim', type=int, default=32, help="encoder LSTM dimensions")
    parser.add_argument('--decoder_dim', type=int, default=32, help="decoder LSTM dimensions")
    parser.add_argument('--embedding_dim', type=int, default=16, help="input embedding dimensions")
    parser.add_argument('--attention_dim', type=int, default=32, help="attention embedding dimensions")
    parser.add_argument('--ip_dim', type=int, default=2, help="input features")
    parser.add_argument('--op_dim', type=int, default=2, help="output features")

    # Generative Parameters
    parser.add_argument('--noise_type', type=str, default="gaussian", choices=["gaussian", "uniform"], help="noise type for generator")
    parser.add_argument('--noise_mix_type', type=str, choices=['ped', 'sample'], default='sample', help='noise mixed for each ped vs. entire sample')
    parser.add_argument('--noise_dim', type=int, default=16, help="noise dim")
    parser.add_argument('--best_k', type=int, default=5, help="k for variety loss")
    parser.add_argument('--l', type=float, default=0, help="lambda value for diversity loss")
    parser.add_argument('--num_traj', type=int, default=20, help="number of trajectories to generate per pedestrian during evaluation")
    parser.add_argument('--d_spatial', action='store_true', help="discriminator takes into account spatial attention")
    parser.add_argument('--l2_loss_weight', type=float, default=0, help="l2 loss weight")
    parser.add_argument('--d_type', choices=['global','local'], default='global', help="discriminator input is observation+prediction in global, only prediction in local")
    parser.add_argument('--d_model_type', choices=['fc', 'lstm'], default='lstm', help="type of discriminator model")
    parser.add_argument('--d_domain', action='store_true', help="discriminator learns its own domain, if false discriminator adopts the generator's learned domain")

    # Domain Parameters
    parser.add_argument('--domain_parameter', type=float, default=2, help="parameter for domain initialization")
    parser.add_argument('--delta_bearing', type=float, default=30, help="relative bearing discretization")
    parser.add_argument('--delta_heading', type=float, default=30, help="relative heading discretization")

    # Training Parameters
    parser.add_argument('--batch_size', type=int, default=8, help="training batch size")
    parser.add_argument('--eval_batch_size', type=int, help="evaluation batch size")
    parser.add_argument('--num_epochs', type=int, default=200, help="number of training epochs")
    parser.add_argument('--lr', type=float, default=0.00005, help="learning rate for training")
    parser.add_argument('--lr_g', type=float, default=0.00005, help="learning rate for training generator")
    parser.add_argument('--lr_d', type=float, default=0.00005, help="learning rate for training discriminator")
    parser.add_argument('--clip', type=float, help="clipping value for clipping gradients")
    parser.add_argument('--gpu_id', type=int, help="gpu id if cuda to be used")
    parser.add_argument('--scheduler',action='store_true',help='use scheduler')
    parser.add_argument('--train_saved', action='store_true', help='train a saved model further')
    # Testing Parameters
    parser.add_argument('--test_only', action='store_true', help="only evaluate a trained model")
    args = parser.parse_args()
    return args
