import os
import pdb
import argparse
from solver import CV_gaitGAN
from mylog import myLog
from torch.backends import cudnn
parser = argparse.ArgumentParser()

# model configuration.
# data related
parser.add_argument('--dataset_dir', dest='dataset_dir', default='./data/gei', help='root path to CASIA(B) dataset')
parser.add_argument('--image_size', dest='image_size', type=int, default=64, help='size of input image')
parser.add_argument('--cond_dim', dest='cond_dim', type=int, default=11, help='length of condition vector')
# parser.add_argument('--st_dim', dest='st_dim', type=int, default=3, help='length of condition vector')
parser.add_argument('--ag_dim', dest='ag_dim', type=int, default=11, help='length of condition vector')
# model related parameters
parser.add_argument('--gf_dim', dest='gf_dim', type=int, default=64, help='number of filters for generator')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=100, help='number of images in a batch')
parser.add_argument('--df_dim', dest='df_dim', type=int, default=64, help='number of filters for discriminator')
parser.add_argument('--ed_dim', dest='ed_dim', type=int, default=64, help='number of dimensions for embedding')
parser.add_argument('--repeat_num', dest='repeat_num', type=int, default=3, help='number of residual blocks')

# balance parameters
parser.add_argument('--lambda_rec', dest='lambda_rec', type=float, default=10., help='balance parameter for reconstruction loss')
parser.add_argument('--lambda_cls', dest='lambda_cls', type=float, default=1., help='balance parameter for classification loss')
parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=0, help='balance parameter for gradient penalty')
parser.add_argument('--lambda_triplet', dest='lambda_triplet', type=float, default=0, help='balance parameter for gradient penalty')
# if-conditions
parser.add_argument('--is_train', dest='is_train', action='store_false', help='training if specify it')
parser.add_argument('--is_gp', dest='is_gp', default=True, help='training if specify it')
# training related
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', dest='beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--g_lr', dest='g_lr', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--d_lr', dest='d_lr', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument('--id_lr', dest='id_lr', type=float, default=0.0001, help='learning rate for id loss')
parser.add_argument('--lr_update_step', dest='lr_update_step', type=int, default=1000, help='update lr every LR_UPDATE_STEP')
parser.add_argument('--num_iters_decay', dest='num_iters_decay',type=int, default=100000, help='number of iterations for decay lr')
parser.add_argument('--num_iters', dest='num_iters',type=int, default=200000, help='number of iterations for training')
parser.add_argument('--resume_iters', dest='resume_iters',type=int, default=200000, help='Resuming training from RESUME_ITERS')
parser.add_argument('--num_critic', dest='num_critic',type=int, default=5, help='number of generator updates for each discriminator update')

# testing related
parser.add_argument('--dst_angle', dest='dst_angle', type=int, default=54, help='target angle in testing.')
parser.add_argument('--test_dir', dest='test_dir', type=str, default='./generated/20181221', help='target angle in testing.')

# configuration for logs
parser.add_argument('--visual_step', dest='visual_step', type=int, default=1000, help='number of steps for visualizing')
parser.add_argument('--sample_step', dest='sample_step', type=int, default=1000, help='number of steps for sampling')
parser.add_argument('--model_ckpt_step', dest='model_ckpt_step', type=int, default=1000, help='number of steps for saving model')

# directories
parser.add_argument('--sample_dir', dest='sample_dir', type=str, default='./sample/20181221', help='path to store samples in training phase')
parser.add_argument('--model_dir', dest='model_dir', type=str, default='./checkpoint/20181221', help='path to store samples in testing phase')
parser.add_argument('--log_dir', dest='log_dir', type=str, default='./logs/20181221', help='path to logs')
parser.add_argument('--loss_dir', dest='loss_dir', type=str, default='./loss/20181221', help='path to store losses')

args = parser.parse_args()

def main():
    # create directories
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.loss_dir):
        os.makedirs(args.loss_dir)

    # create logger
    logger = myLog(args.log_dir)
    # create model
    model = CV_gaitGAN(args, logger)
    # pdb.set_trace()
    if args.is_train:
        model.train()
    else:
        print('testing.......................')
        if not os.path.exists(args.test_dir):
            os.makedirs(args.test_dir)
        model.test()


if __name__ == "__main__":
    main()
