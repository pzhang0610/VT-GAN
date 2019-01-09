from model import *
from dataset import CasiaGait, CasiaGaitVal, CasiaGaitTest
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import time
from torchvision.utils import save_image
import os
from scipy.misc import imsave
import pdb
class gaitStarGAN(nn.Module):
    """Solver for gaitStarGAN"""
    def __init__(self, args, logger):
        super(gaitStarGAN, self).__init__()
        self.dataset_dir = args.dataset_dir

        # Model configurations.
        self.batch_size = args.batch_size
        self.gf_dim = args.gf_dim
        self.df_dim = args.df_dim
        self.ag_dim = args.ag_dim
        self.g_repeat_num = args.repeat_num
        self.d_repeat_num = args.repeat_num

        self.image_size = args.image_size
        self.cond_dim = args.cond_dim

        # if_conditions parameters in model
        self.is_gp = args.is_gp

        # balance parameters
        self.lambda_rec = args.lambda_rec
        self.lambda_cls = args.lambda_cls
        self.lambda_gp  = args.lambda_gp
        self.lambda_triplet = args.lambda_triplet

        # Configurations for Training.
        self.is_train = args.is_train
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        self.lr_update_step = args.lr_update_step
        self.num_iter_decay = args.num_iters_decay
        self.num_iters = args.num_iters
        self.resume_iters = args.resume_iters
        self.num_critic = args.num_critic

        self.dst_angle = args.dst_angle
        self.test_dir = args.test_dir

        # configuration for logs
        self.logger = logger
        self.visual_step = args.visual_step
        self.sample_step = args.sample_step
        self.model_ckpt_step = args.model_ckpt_step
        # Other configurations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration for Directories
        self.sample_dir = args.sample_dir
        self.model_dir = args.model_dir
        self.log_dir = args.log_dir
        self.loss_dir = args.loss_dir

        # Build model
        self.build()


    def build(self):
        self.logger.add_log("{} Building model...".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), is_print=True)
        self.G = Generator(self.gf_dim, self.cond_dim, self.g_repeat_num)
        if self.is_train:
            self.D = Discriminator(self.image_size, self.df_dim, self.ag_dim, self.d_repeat_num)

        if self.is_train:
            self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, (self.beta1, self.beta2))
            self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, (self.beta1, self.beta2))

        self.G.to(self.device)
        self.init_weights(self.G, init_type='normal')
        if self.is_train:
            self.D.to(self.device)
            self.init_weights(self.D, init_type='normal')

        if self.is_train:
            self.l1_loss = nn.L1Loss(size_average=True)
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.cls_loss = nn.CrossEntropyLoss()
            self.triplet_loss = nn.TripletMarginLoss(margin=0.8, p=2)

    # def cls_loss(self, logit, target):
    #     return F.cross_entropy(logit, target)

    def set_input(self, data, src_label, dst_label):
        """set input data to the model."""
        self.real_A = data[:, :1, :].to(self.device)
        self.real_B = data[:, 1:2, :].to(self.device)
        self.label_A = src_label.to(self.device)
        self.label_B = dst_label.to(self.device)
        self.label_ag_B = torch.argmax(self.label_B, dim=1).to(self.device)
        # self.label_st_B = torch.argmax(self.label_B[:, 11:], dim=1).to(self.device)
        if self.is_train:
            self.neg = data[:, 2:3, :, :].to(self.device)
            self.pos = data[:, 3:, :, :].to(self.device)

    def foward_G(self):
        self.fake_B = self.G(self.real_A, self.label_B)
        # self.fake_A = self.G(self.fake_B, self.label_A)

    def backward_G(self):
        dis_logit, cls_angle_logit = self.D(self.fake_B)
        self.g_loss_fake = self.bce_loss(dis_logit, torch.ones_like(dis_logit))
        self.g_loss_cls_angle = self.cls_loss(cls_angle_logit, self.label_ag_B)
        # self.g_loss_cls_state = self.cls_loss(cls_state_logit, self.label_st_B)
        self.g_loss_l1 = self.l1_loss(self.fake_B, self.real_B)
        # self.g_loss_rec = self.l1_loss(self.fake_A, self.real_A)

        self.g_loss_triplet = self.triplet_loss(self.fake_B, self.pos, self.neg)

        self.g_loss = self.g_loss_fake + self.lambda_cls * self.g_loss_cls_angle + self.lambda_rec *self.g_loss_l1 + self.lambda_triplet * self.g_loss_triplet

        self.g_loss.backward()


    def backward_D(self):
        # loss for real images
        dis_logit, cls_angle_logit = self.D(self.real_B)
        self.d_loss_real = self.bce_loss(dis_logit, torch.ones_like(dis_logit))
        self.d_loss_cls_angle = self.cls_loss(cls_angle_logit, self.label_ag_B)
        # self.d_loss_cls_state = self.cls_loss(cls_state_logit, self.label_st_B)

        # loss for fake images
        dis_logit, cls_state_logit = self.D(self.fake_B.detach())
        self.d_loss_fake = self.bce_loss(dis_logit, torch.zeros_like(dis_logit))

        # gradient penalty
        if self.is_gp:
            alpha = torch.rand(self.real_B.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * self.real_B.data + (1 - alpha) * self.fake_B.data).requires_grad_(True)
            dis_logit, _ = self.D(x_hat)
            self.d_loss_gp = self.gradient_penalty(dis_logit, x_hat)
            self.d_loss = self.d_loss_real + self.d_loss_fake + self.lambda_gp * self.d_loss_gp + \
                self.lambda_cls * self.d_loss_cls_angle
        else:
            self.d_loss = self.d_loss_real + self.d_loss_fake + self.lambda_cls * self.d_loss_cls_angle

        self.d_loss.backward()

    def init_weights(self, net, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Linear') != -1 and hasattr(m, 'weight'):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('Initialization method {} is not found...'.format(init_type))
            elif classname.find("BatchNorm2d") != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
        self.logger.add_log("{} Initialize network with {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), init_type), is_print=True)
        net.apply(init_func)

    def optimize_params(self, i):
        self.foward_G()
        # update D
        self.set_requires_grad(self.D, True)
        self.d_optimizer.zero_grad()
        self.backward_D()
        self.d_optimizer.step()

        # update G
        if i % self.num_critic == 0:
            self.set_requires_grad(self.D, False)
            self.g_optimizer.zero_grad()
            self.backward_G()
            self.g_optimizer.step()

    def print_network(self, model):
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        print(model)
        print("The number of params is {}".format(num_params))

    def update_lr(self, g_lr, d_lr):
        for params in self.g_optimizer.param_groups:
            params['lr'] = g_lr
        for params in self.d_optimizer.param_groups:
            params['lr'] = d_lr

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def denorm(self, x):
        out = (x + 1.)/2.
        return out.clamp_(0, 1)

    def name(self):
        return 'gaitStarGAN'

    def train(self):
        # input data
        dataLoader = CasiaGait(self.dataset_dir, self.batch_size, self.image_size)

        # Learning rate for decay
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Training from scratch or resume training
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore(self.resume_iters, is_train=self.is_train)
        else:
            self.g_loss_logs = {}
            self.d_loss_logs = {}
        # Start training
        self.logger.add_log('{} Starting training from Iter {}...'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                                          start_iters), is_print=True)
        fig = plt.figure(1)
        plt.ion()
        for it in range(start_iters, self.num_iters):
            # get a batch of data
            batch_imgs, src_labels, dst_labels, spl_name = dataLoader.get_batch()
            self.set_input(batch_imgs, src_labels, dst_labels)
            iter_start_time = time.time()
            self.optimize_params(it)
            batch_running_time = time.time() - iter_start_time

            self.logger.add_log("{} Iteration [{}/{}] g_loss: {:.6f}, d_loss: {:.6f}, elapse: {:.4f} seconds".
                                format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), it, self.num_iters,
                                       self.g_loss.item(), self.d_loss.item(), batch_running_time), is_print=True)

            # Visualize losses
            if (it + 1) % self.visual_step == 0:
                self.g_loss_logs[(it+1)//self.visual_step] = self.g_loss.item()
                self.d_loss_logs[(it+1)//self.visual_step] = self.d_loss.item()
                # save loss
                self.save_losses(it)
                # visualize loss
                plt.cla()
                ax1 = fig.add_subplot(311)
                ax1.plot(list(self.g_loss_logs.keys()), list(self.g_loss_logs.values()))
                ax1.set_xlabel('Iteration (k)')
                ax1.set_ylabel('Loss')
                ax2 = fig.add_subplot(312)
                ax2.plot(list(self.d_loss_logs.keys()), list(self.d_loss_logs.values()))
                ax2.set_xlabel('Iteration (k)')
                ax2.set_ylabel('Loss')
                ax3 = fig.add_subplot(313)
                ax3.plot(list(self.g_loss_logs.keys()), list(self.g_loss_logs.values()), 'r-', list(self.d_loss_logs.keys()), list(self.d_loss_logs.values()), 'g-')
                ax3.set_xlabel('Iteration (k)')
                ax3.set_ylabel('Loss')
                ax3.legend(['G', 'D'])
                plt.pause(0.001)

            # Sampling
            if (it + 1) % self.sample_step == 0:
                self.eval(it)

            # Saving model checkpoints
            if (it + 1) % self.model_ckpt_step == 0:
                self.save_model(self.model_dir, it)

            # Decay learning rate
            if (it + 1) % self.lr_update_step == 0 and (it + 1) > (self.num_iters - self.num_iter_decay):
                g_lr -= (self.g_lr/float(self.num_iter_decay))
                d_lr -= (self.d_lr/float(self.num_iter_decay))
                self.update_lr(g_lr, d_lr)
        plt.ioff()
        plt.show()

    def eval(self, it):
        self.logger.add_log("{} Performing sampling...".format(datetime.now().strftime("%Y-%m-%d %H:%M:S")), is_print=True)
        dataLoader = CasiaGaitVal(self.dataset_dir, self.batch_size, self.image_size)
        batch_imgs, src_labels, dst_labels, src_names, dst_names = dataLoader.get_batch()
        with torch.no_grad():
            self.set_input(batch_imgs, src_labels, dst_labels)
            fake_B = self.G(self.real_A, self.label_B)
            imgs = self.denorm(fake_B.data.cpu())
            sample_path_fake = os.path.join(self.sample_dir, 'sample-fake-{}.jpg'.format(it + 1))
            save_image(imgs, sample_path_fake, nrow=10)

            real_B = self.denorm(self.real_B.data.cpu())
            sample_path_real = os.path.join(self.sample_dir, 'sample-real-{}.jpg'.format(it + 1))
            save_image(real_B, sample_path_real, nrow=10)

    def test(self):
        self.logger.add_log("{} Performing sampling for testing...".format(datetime.now().strftime("%Y-%m-%d %H:%M:S")), is_print=True)
        self.restore(self.resume_iters, is_train=self.is_train)
        dataLoader = CasiaGaitTest(self.dataset_dir, self.image_size)
        for p in range(1, 125):
            for cond in range(10):
                for angle in range(11):
                    batch_item = dataLoader.get_item(p, cond, angle, self.dst_angle)
                    if isinstance(batch_item, tuple):
                        src_image = batch_item[0]
                        src_name = batch_item[1]
                        dst_label = batch_item[2]
                    else:
                        continue
                    with torch.no_grad():
                        self.real_A = src_image.to(self.device)
                        self.label_B = dst_label.to(self.device)
                        fake_B = self.G(self.real_A, self.label_B)
                        img = self.denorm(fake_B.data.cpu())
                    save_path = os.path.join(self.test_dir, '%03d'%self.dst_angle)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_path_name = os.path.join(self.test_dir, '%03d'%self.dst_angle, src_name)
                    # save_image(img, save_path_name,nrow=1)
                    imsave(save_path_name, np.squeeze(np.transpose(img.numpy(), (0, 2, 3, 1))))

    def save_model(self, model_dir, it):
        self.logger.add_log("{} Saving model on step {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:S"),it + 1), is_print=True)
        model_path = os.path.join(model_dir, self.name() + '-{}.pth'.format(it))
        torch.save({'g_static_dict': self.G.state_dict(),
                    'd_static_dict': self.D.state_dict(),
                    'g_optimizer_static_dict': self.g_optimizer.state_dict(),
                    'd_optimizer_static_dict': self.d_optimizer.state_dict()}, model_path)

    def restore(self, resume_iter, is_train):
        self.logger.add_log("{} Resuming model from step {}...".format(datetime.now().strftime("%Y-%m-%d %H:%M:S"), resume_iter + 1), is_print=True)
        model_path = os.path.join(self.model_dir, self.name() + '-{}.pth'.format(resume_iter))
        checkpoint = torch.load(model_path)
        if is_train:
            self.G.load_state_dict(checkpoint['g_static_dict'])
            self.D.load_state_dict(checkpoint['d_static_dict'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_static_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_static_dict'])
            self.G.train()
            self.D.train()

            loss_path = os.path.join(self.loss_dir, self.name() + '-losses-{}.npy'.format(resume_iter))
            losses = np.load(loss_path).item()
            self.g_loss_logs = losses['g_losses']
            self.d_loss_logs = losses['d_losses']
        else:
            self.G.load_state_dict(checkpoint['g_static_dict'], strict=False)
            # self.G.eval()
            # for i in self.G.named_parameters():
            #     print(i)


    def save_losses(self, it):
        loss_path = os.path.join(self.loss_dir, self.name() + '-losses-{}.npy'.format(it))
        losses = {'g_losses': self.g_loss_logs, 'd_losses': self.d_loss_logs}
        np.save(loss_path, losses)


class CV_gaitGAN(nn.Module):
    """Solver for gaitStarGAN"""
    def __init__(self, args, logger):
        super(CV_gaitGAN, self).__init__()
        self.dataset_dir = args.dataset_dir

        # Model configurations.
        self.batch_size = args.batch_size
        self.gf_dim = args.gf_dim
        self.df_dim = args.df_dim
        self.ed_dim = args.ed_dim
        self.ag_dim = args.ag_dim
        self.g_repeat_num = args.repeat_num
        self.d_repeat_num = args.repeat_num

        self.image_size = args.image_size
        self.cond_dim = args.cond_dim

        # if_conditions parameters in model
        self.is_gp = args.is_gp

        # balance parameters
        self.lambda_rec = args.lambda_rec
        self.lambda_cls = args.lambda_cls
        self.lambda_gp  = args.lambda_gp
        self.lambda_triplet = args.lambda_triplet

        # Configurations for Training.
        self.is_train = args.is_train
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        self.id_lr = args.id_lr
        self.lr_update_step = args.lr_update_step
        self.num_iter_decay = args.num_iters_decay
        self.num_iters = args.num_iters
        self.resume_iters = args.resume_iters
        self.num_critic = args.num_critic

        self.dst_angle = args.dst_angle
        self.test_dir = args.test_dir

        # configuration for logs
        self.logger = logger
        self.visual_step = args.visual_step
        self.sample_step = args.sample_step
        self.model_ckpt_step = args.model_ckpt_step
        # Other configurations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration for Directories
        self.sample_dir = args.sample_dir
        self.model_dir = args.model_dir
        self.log_dir = args.log_dir
        self.loss_dir = args.loss_dir

        # Build model
        self.build()


    def build(self):
        self.logger.add_log("{} Building model...".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), is_print=True)
        self.G = Generator(self.gf_dim, self.cond_dim, self.g_repeat_num)
        if self.is_train:
            self.D = Discriminator(self.image_size, self.df_dim, self.ag_dim, self.d_repeat_num)
            self.ID = SiaNet(ed_dim=self.ed_dim)

        if self.is_train:
            self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, (self.beta1, self.beta2))
            self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, (self.beta1, self.beta2))
            self.id_optimizer = torch.optim.Adam(self.ID.parameters(), self.id_lr, (self.beta1, self.beta2))

        self.G.to(self.device)
        self.init_weights(self.G, init_type='normal')
        if self.is_train:
            self.D.to(self.device)
            self.init_weights(self.D, init_type='normal')
            self.ID.to(self.device)
            self.init_weights(self.ID, init_type='normal')

        if self.is_train:
            self.l1_loss = nn.L1Loss(size_average=True)
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.cls_loss = nn.CrossEntropyLoss()
            self.triplet_loss = nn.TripletMarginLoss(margin=0.8, p=2)

    # def cls_loss(self, logit, target):
    #     return F.cross_entropy(logit, target)

    def set_input(self, data, src_label, dst_label):
        """set input data to the model."""
        self.real_A = data[:, :1, :].to(self.device)
        self.real_B = data[:, 1:2, :].to(self.device)
        self.label_A = src_label.to(self.device)
        self.label_B = dst_label.to(self.device)
        self.label_ag_B = torch.argmax(self.label_B, dim=1).to(self.device)
        # self.label_st_B = torch.argmax(self.label_B[:, 11:], dim=1).to(self.device)
        if self.is_train:
            self.neg = data[:, 2:3, :, :].to(self.device)
            self.pos = data[:, 3:, :, :].to(self.device)

    def foward_G(self):
        self.fake_B = self.G(self.real_A, self.label_B)
        # self.fake_A = self.G(self.fake_B, self.label_A)

    def backward_G(self):
        dis_logit, cls_angle_logit = self.D(self.fake_B)
        embedding = self.ID(self.fake_B)
        neg_embedding = self.ID(self.neg)
        pos_embedding = self.ID(self.pos)

        g_anc_embedding = F.normalize(embedding, p=2)
        g_neg_embedding = F.normalize(neg_embedding, p=2)
        g_pos_embedding = F.normalize(pos_embedding, p=2)

        self.g_loss_fake = self.bce_loss(dis_logit, torch.ones_like(dis_logit))
        self.g_loss_cls_angle = self.cls_loss(cls_angle_logit, self.label_ag_B)
        # self.g_loss_cls_state = self.cls_loss(cls_state_logit, self.label_st_B)
        self.g_loss_l1 = self.l1_loss(self.fake_B, self.real_B)
        # self.g_loss_rec = self.l1_loss(self.fake_A, self.real_A)

        self.g_loss_triplet = self.triplet_loss(g_anc_embedding, g_pos_embedding, g_neg_embedding)

        self.g_loss = self.g_loss_fake + self.lambda_cls * self.g_loss_cls_angle + self.lambda_rec *self.g_loss_l1 + self.lambda_triplet * self.g_loss_triplet

        self.g_loss.backward()


    def backward_D(self):
        # loss for real images
        dis_logit, cls_angle_logit = self.D(self.real_B)
        self.d_loss_real = self.bce_loss(dis_logit, torch.ones_like(dis_logit))
        self.d_loss_cls_angle = self.cls_loss(cls_angle_logit, self.label_ag_B)
        # self.d_loss_cls_state = self.cls_loss(cls_state_logit, self.label_st_B)

        # loss for fake images
        dis_logit, cls_state_logit = self.D(self.fake_B.detach())
        self.d_loss_fake = self.bce_loss(dis_logit, torch.zeros_like(dis_logit))

        # gradient penalty
        if self.is_gp:
            alpha = torch.rand(self.real_B.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * self.real_B.data + (1 - alpha) * self.fake_B.data).requires_grad_(True)
            dis_logit, _ = self.D(x_hat)
            self.d_loss_gp = self.gradient_penalty(dis_logit, x_hat)
            self.d_loss = self.d_loss_real + self.d_loss_fake + self.lambda_gp * self.d_loss_gp + \
                self.lambda_cls * self.d_loss_cls_angle
        else:
            self.d_loss = self.d_loss_real + self.d_loss_fake + self.lambda_cls * self.d_loss_cls_angle

        self.d_loss.backward()

    def backward_ID(self):
        embedding = self.ID(self.fake_B.detach())
        anc_embedding = self.ID(self.real_B.detach())
        neg_embedding = self.ID(self.neg.detach())
        pos_embedding = self.ID(self.pos.detach())

        id_fake_embedding = F.normalize(embedding, p=2)
        id_anc_embedding = F.normalize(anc_embedding, p=2)
        id_neg_embedding = F.normalize(neg_embedding, p=2)
        id_pos_embedding = F.normalize(pos_embedding, p=2)

        # self.id_triplet_loss = 0.5 * (self.triplet_loss(id_anc_embedding, id_pos_embedding, id_neg_embedding)
        #                               + self.triplet_loss(id_fake_embedding, id_pos_embedding, id_neg_embedding))
        self.id_triplet_loss = self.triplet_loss(id_anc_embedding, id_pos_embedding, id_neg_embedding)
        self.id_triplet_loss.backward()

    def init_weights(self, net, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Linear') != -1 and hasattr(m, 'weight'):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('Initialization method {} is not found...'.format(init_type))
            elif classname.find("BatchNorm2d") != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
        self.logger.add_log("{} Initialize network with {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), init_type), is_print=True)
        net.apply(init_func)

    def optimize_params(self, i):
        self.foward_G()
        # update D
        self.set_requires_grad(self.D, True)
        self.d_optimizer.zero_grad()
        self.backward_D()
        self.d_optimizer.step()
        # pdb.set_trace()

        # update ID
        self.set_requires_grad(self.ID, True)
        self.id_optimizer.zero_grad()
        self.backward_ID()
        self.id_optimizer.step()

        # update G
        if i % self.num_critic == 0:
            self.set_requires_grad([self.D, self.ID], False)
            self.g_optimizer.zero_grad()
            self.backward_G()
            self.g_optimizer.step()

    def print_network(self, model):
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        print(model)
        print("The number of params is {}".format(num_params))

    def update_lr(self, g_lr, d_lr, id_lr):
        for params in self.g_optimizer.param_groups:
            params['lr'] = g_lr
        for params in self.d_optimizer.param_groups:
            params['lr'] = d_lr
        for params in self.id_optimizer.param_groups:
            params['lr'] = id_lr

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def denorm(self, x):
        out = (x + 1.)/2.
        return out.clamp_(0, 1)

    def name(self):
        return 'gaitStarGAN'

    def train(self):
        # input data
        dataLoader = CasiaGait(self.dataset_dir, self.batch_size, self.image_size)

        # Learning rate for decay
        g_lr = self.g_lr
        d_lr = self.d_lr
        id_lr = self.id_lr

        # Training from scratch or resume training
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters + 1
            self.restore(self.resume_iters, is_train=self.is_train)
        else:
            self.g_loss_logs = {}
            self.d_loss_logs = {}
            self.id_loss_logs = {}
        # Start training
        self.logger.add_log('{} Starting training from Iter {}...'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                                          start_iters), is_print=True)
        fig = plt.figure(1)
        plt.ion()
        for it in range(start_iters, self.num_iters):
            # get a batch of data
            batch_imgs, src_labels, dst_labels, spl_name = dataLoader.get_batch()
            self.set_input(batch_imgs, src_labels, dst_labels)
            iter_start_time = time.time()
            self.optimize_params(it)
            batch_running_time = time.time() - iter_start_time


            self.logger.add_log("{} Iteration [{}/{}] g_loss: {:.6f}, d_loss: {:.6f}, id_loss: {:,.6f}, elapse: {:.4f} seconds".
                                    format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), it, self.num_iters,
                                        self.g_loss.item(), self.d_loss.item(), self.id_triplet_loss.item(), batch_running_time), is_print=True)

            # Visualize losses
            if (it + 1) % self.visual_step == 0:
                self.g_loss_logs[(it+1)//self.visual_step] = self.g_loss.item()
                self.d_loss_logs[(it+1)//self.visual_step] = self.d_loss.item()
                self.id_loss_logs[(it+1)//self.visual_step] = self.id_triplet_loss.item()
                # save loss
                self.save_losses(it)
                # visualize loss
                plt.cla()
                ax1 = fig.add_subplot(221)
                ax1.plot(list(self.g_loss_logs.keys()), list(self.g_loss_logs.values()))
                ax1.set_xlabel('Iteration (k)')
                ax1.set_ylabel('Loss')
                ax2 = fig.add_subplot(222)
                ax2.plot(list(self.d_loss_logs.keys()), list(self.d_loss_logs.values()))
                ax2.set_xlabel('Iteration (k)')
                ax2.set_ylabel('Loss')
                ax3 = fig.add_subplot(223)
                ax3.plot(list(self.id_loss_logs.keys()), list(self.id_loss_logs.values()))
                ax3.set_xlabel('Iteration (k)')
                ax3.set_ylabel('Loss')
                ax4 = fig.add_subplot(224)
                ax4.plot(list(self.g_loss_logs.keys()), list(self.g_loss_logs.values()), 'r-', list(self.d_loss_logs.keys()), list(self.d_loss_logs.values()), 'g-')
                ax4.set_xlabel('Iteration (k)')
                ax4.set_ylabel('Loss')
                ax4.legend(['G', 'D'])
                plt.pause(0.001)

            # Sampling
            if (it + 1) % self.sample_step == 0:
                self.eval(it)

            # Saving model checkpoints
            if (it + 1) % self.model_ckpt_step == 0:
                self.save_model(self.model_dir, it)

            # Decay learning rate
            if (it + 1) % self.lr_update_step == 0 and (it + 1) > (self.num_iters - self.num_iter_decay):
                g_lr -= (self.g_lr/float(self.num_iter_decay))
                d_lr -= (self.d_lr/float(self.num_iter_decay))
                id_lr -= (self.id_lr/float(self.num_iter_decay))

                self.update_lr(g_lr, d_lr, id_lr)
        plt.ioff()
        plt.show()

    def eval(self, it):
        self.logger.add_log("{} Performing sampling...".format(datetime.now().strftime("%Y-%m-%d %H:%M:S")), is_print=True)
        dataLoader = CasiaGaitVal(self.dataset_dir, self.batch_size, self.image_size)
        batch_imgs, src_labels, dst_labels, src_names, dst_names = dataLoader.get_batch()
        with torch.no_grad():
            self.set_input(batch_imgs, src_labels, dst_labels)
            fake_B = self.G(self.real_A, self.label_B)
            imgs = self.denorm(fake_B.data.cpu())
            sample_path_fake = os.path.join(self.sample_dir, 'sample-fake-{}.jpg'.format(it + 1))
            save_image(imgs, sample_path_fake, nrow=10)

            real_B = self.denorm(self.real_B.data.cpu())
            sample_path_real = os.path.join(self.sample_dir, 'sample-real-{}.jpg'.format(it + 1))
            save_image(real_B, sample_path_real, nrow=10)

    def test(self):
        self.logger.add_log("{} Performing sampling for testing...".format(datetime.now().strftime("%Y-%m-%d %H:%M:S")), is_print=True)
        self.restore(self.resume_iters, is_train=self.is_train)
        dataLoader = CasiaGaitTest(self.dataset_dir, self.image_size)
        for p in range(1, 125):
            for cond in range(10):
                for angle in range(11):
                    batch_item = dataLoader.get_item(p, cond, angle, self.dst_angle)
                    if isinstance(batch_item, tuple):
                        src_image = batch_item[0]
                        src_name = batch_item[1]
                        dst_label = batch_item[2]
                    else:
                        continue
                    with torch.no_grad():
                        self.real_A = src_image.to(self.device)
                        self.label_B = dst_label.to(self.device)
                        fake_B = self.G(self.real_A, self.label_B)
                        # embedding = F.normalize(self.ID(fake_B), p=2)
                        img = self.denorm(fake_B.data.cpu())
                    save_path = os.path.join(self.test_dir,'imgs', '%03d'%self.dst_angle)
                    # embedding_path = os.path.join(self.test_dir, 'embedding', '%03d'%self.dst_angle)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                        # os.makedirs(embedding_path)
                    save_path_name = os.path.join(save_path, src_name)
                    # embedding_path_name = os.path.join(embedding_path, '%03d'%self.dst_angle, src_name[:-4]+'.npy')
                    # save_image(img, save_path_name,nrow=1)
                    imsave(save_path_name, np.squeeze(np.transpose(img.numpy(), (0, 2, 3, 1))))
                    # np.save(embedding_path_name, embedding)


    def save_model(self, model_dir, it):
        self.logger.add_log("{} Saving model on step {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:S"),it + 1), is_print=True)
        model_path = os.path.join(model_dir, self.name() + '-{}.pth'.format(it))
        torch.save({'g_static_dict': self.G.state_dict(),
                    'd_static_dict': self.D.state_dict(),
                    'id_static_dict': self.ID.state_dict(),
                    'g_optimizer_static_dict': self.g_optimizer.state_dict(),
                    'd_optimizer_static_dict': self.d_optimizer.state_dict(),
                    'id_optimizer_static_dict': self.id_optimizer.state_dict()}, model_path)

    def restore(self, resume_iter, is_train):
        self.logger.add_log("{} Resuming model from step {}...".format(datetime.now().strftime("%Y-%m-%d %H:%M:S"), resume_iter + 1), is_print=True)
        model_path = os.path.join(self.model_dir, self.name() + '-{}.pth'.format(resume_iter))
        checkpoint = torch.load(model_path)
        if is_train:
            self.G.load_state_dict(checkpoint['g_static_dict'])
            self.D.load_state_dict(checkpoint['d_static_dict'])
            self.ID.load_state_dict(checkpoint['id_static_dict'])

            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_static_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_static_dict'])
            self.id_optimizer.load_state_dict(checkpoint['id_optimizer_static_dict'])
            self.G.train()
            self.D.train()
            self.ID.train()

            loss_path = os.path.join(self.loss_dir, self.name() + '-losses-{}.npy'.format(resume_iter))
            losses = np.load(loss_path).item()
            self.g_loss_logs = losses['g_losses']
            self.d_loss_logs = losses['d_losses']
            self.id_loss_logs = losses['id_losses']
        else:
            self.G.load_state_dict(checkpoint['g_static_dict'], strict=False)
            # self.ID.load_state_dict(checkpoint['id_static_dict'], strict=False)
            # self.G.eval()
            # for i in self.G.named_parameters():
            #     print(i)


    def save_losses(self, it):
        loss_path = os.path.join(self.loss_dir, self.name() + '-losses-{}.npy'.format(it))
        losses = {'g_losses': self.g_loss_logs, 'd_losses': self.d_loss_logs, 'id_losses': self.id_loss_logs}
        np.save(loss_path, losses)


