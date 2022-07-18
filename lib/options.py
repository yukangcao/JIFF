import argparse
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        g_data = parser.add_argument_group('Data')  #建立参数解析工具 
        g_data.add_argument('--dataroot', type=str, default='./data',
                            help='path to images (data folder)') #导入命令行输入数据(dataroot)
        g_data.add_argument('--dataroot_512', type=str, default='./data',
                            help='path to 512 images (data folder)')

        g_data.add_argument('--loadSize', type=int, default=512, help='load size of input image')  #导入命令行数据（loadsize）

        # Experiment related
        g_exp = parser.add_argument_group('Experiment') #(experiment)
        g_exp.add_argument('--name', type=str, default='example',
                           help='name of the experiment. It decides where to store samples and models') # name
        g_exp.add_argument('--debug', action='store_true', help='debug mode or not') # debug

        g_exp.add_argument('--num_views', type=int, default=1, help='How many views to use for multiview network.') #view数量
        g_exp.add_argument('--random_multiview', action='store_true', help='Select random multiview combination.')  # random multiview

        # Training related
        g_train = parser.add_argument_group('Training') #（Training）
        g_train.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda') #gpuid
        g_train.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')

        g_train.add_argument('--num_threads', default=1, type=int, help='# sthreads for loading data') #输入data的线程数
        g_train.add_argument('--serial_batches', action='store_true',
                             help='if true, takes images in order to make batches, otherwise takes them randomly') #制作minibatch
        g_train.add_argument('--pin_memory', action='store_true', help='pin_memory')
        
        g_train.add_argument('--batch_size', type=int, default=2, help='input batch size') #batch size
        g_train.add_argument('--learning_rate', type=float, default=1e-3, help='adam learning rate') #learning rate
        g_train.add_argument('--learning_rateC', type=float, default=1e-3, help='adam learning rate') 
        g_train.add_argument('--num_epoch', type=int, default=1000, help='num epoch to train') #epoch数

        g_train.add_argument('--freq_plot', type=int, default=10, help='freqency of the error plot') #画error图频率
        g_train.add_argument('--freq_save', type=int, default=50, help='freqency of the save_checkpoints') #保存checkpoint频率
        g_train.add_argument('--freq_save_ply', type=int, default=100, help='freqency of the save ply') #
       
        g_train.add_argument('--no_gen_mesh', action='store_true') 
        g_train.add_argument('--no_num_eval', action='store_true') 
        
        g_train.add_argument('--resume_epoch', type=int, default=-1, help='epoch resuming the training') 
        g_train.add_argument('--continue_train', action='store_true', help='continue training: load the latest model') 

        # Testing related
        g_test = parser.add_argument_group('Testing') 
        g_test.add_argument('--resolution', type=int, default=256, help='# of grid in mesh reconstruction') #resolution
        g_test.add_argument('--test_folder_path', type=str, default=None, help='the folder of test image') #folder path

        # Sampling related
        g_sample = parser.add_argument_group('Sampling') #
        g_sample.add_argument('--sigma', type=float, default=5.0, help='perturbation standard deviation for positions') #perturbation

        g_sample.add_argument('--num_sample_inout', type=int, default=5000, help='# of sampling points') #sample
        g_sample.add_argument('--num_sample_color', type=int, default=0, help='# of sampling points')

        g_sample.add_argument('--z_size', type=float, default=200.0, help='z normalization factor')

        # Model related
        g_model = parser.add_argument_group('Model')
        # General
        g_model.add_argument('--norm', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization') #normalization method
        g_model.add_argument('--norm_color', type=str, default='instance',
                             help='instance normalization or batch normalization or group normalization') #normalization for color

        # hg filter specify
        g_model.add_argument('--num_stack', type=int, default=4, help='# of hourglass') #hourglass结构数量
        g_model.add_argument('--num_hourglass', type=int, default=2, help='# of stacked layer of hourglass')  #
        g_model.add_argument('--skip_hourglass', action='store_true', help='skip connection in hourglass')
        g_model.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        g_model.add_argument('--hourglass_dim', type=int, default='256', help='256 | 512')

        # Classification General
        g_model.add_argument('--mlp_dim', nargs='+', default=[257, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp')
        g_model.add_argument('--mlp_dim_color', nargs='+', default=[513, 1024, 512, 256, 128, 3],
                             type=int, help='# of dimensions of color mlp')
        g_model.add_argument('--mlp_dim_color_fine', nargs='+', default=[513, 1024, 512, 256, 128, 3],
                             type=int, help='# of dimensions of color mlp')
                             
        g_model.add_argument('--mlp_dim_3d_color', nargs='+', default=[128, 1024, 512, 256, 128, 3], type=int, help='# of dimensions of mlp for DeepVoxels 3d branch')
        g_model.add_argument('--mlp_dim_joint_color', nargs='+', default=[0, 1024, 512, 256, 128, 3], type=int, help='# of dimensions of mlp for joint 2d-3d branch')

        g_model.add_argument('--use_tanh', action='store_true',
                             help='using tanh after last conv of image_filter network')
        g_model.add_argument('--mlp_dim_3d', nargs='+', default=[128, 1024, 512, 256, 128, 1], type=int, help='# of dimensions of mlp for DeepVoxels 3d branch')
        g_model.add_argument('--mlp_dim_joint', nargs='+', default=[0, 1024, 512, 256, 128, 1], type=int, help='# of dimensions of mlp for joint 2d-3d branch')

        # for train
        parser.add_argument('--random_flip', action='store_true', help='if random flip')
        parser.add_argument('--random_trans', action='store_true', help='if random flip')
        parser.add_argument('--random_scale', action='store_true', help='if random flip')
        parser.add_argument('--no_residual', action='store_true', help='no skip connection in mlp') #no skip connection
        parser.add_argument('--schedule', type=int, nargs='+', default=[60, 80],
                            help='Decrease learning rate at these epochs.') #decreasing learning rate
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        parser.add_argument('--color_loss_type', type=str, default='l1', help='mse | l1') #loss function for color model

        # for eval
        parser.add_argument('--val_test_error', action='store_true', help='validate errors of test data')  #validate error of testing 
        parser.add_argument('--val_train_error', action='store_true', help='validate errors of train data') #validate error of training
        parser.add_argument('--gen_test_mesh', action='store_true', help='generate test mesh') #生成test—mesh
        parser.add_argument('--gen_train_mesh', action='store_true', help='generate train mesh')
        parser.add_argument('--all_mesh', action='store_true', help='generate meshs from all hourglass output')
        parser.add_argument('--num_gen_mesh_test', type=int, default=1,
                            help='how many meshes to generate during testing') #test_mesh数目

        # path
        parser.add_argument('--checkpoints_path', type=str, default='./checkpoints', help='path to save checkpoints') #save checkpoints
        parser.add_argument('--load_netG_checkpoint_path', type=str, default=None, help='path to save checkpoints') #netG path
        parser.add_argument('--load_netC_checkpoint_path', type=str, default=None, help='path to save checkpoints') #netC path
        parser.add_argument('--load_netC_coarse_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--results_path', type=str, default='./results', help='path to save results ply') # result path
        parser.add_argument('--load_checkpoint_path', type=str, help='path to save results ply') #
        parser.add_argument('--single', type=str, default='', help='single data for training')
        # for single image reconstruction
        parser.add_argument('--mask_path', type=str, help='path for input mask') #path to input mask/single
        parser.add_argument('--img_path', type=str, help='path for input image') #path to input image/single

        # aug
        group_aug = parser.add_argument_group('aug')
        group_aug.add_argument('--aug_alstd', type=float, default=0.0, help='augmentation pca lighting alpha std')
        group_aug.add_argument('--aug_bri', type=float, default=0.0, help='augmentation brightness') #亮度
        group_aug.add_argument('--aug_con', type=float, default=0.0, help='augmentation contrast') #对比度
        group_aug.add_argument('--aug_sat', type=float, default=0.0, help='augmentation saturation') #饱和度
        group_aug.add_argument('--aug_hue', type=float, default=0.0, help='augmentation hue') #hue
        group_aug.add_argument('--aug_blur', type=float, default=0.0, help='augmentation blur') #模糊度

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter) #创建ArgumentParser对象
            parser = self.initialize(parser) #initialization

        self.parser = parser

        return parser.parse_args() #返回到args子类中，增加其属性

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt
