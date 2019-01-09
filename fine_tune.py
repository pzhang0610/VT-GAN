from model import SiaNet
import torch
import torch.nn as nn
from datetime import datetime
import os

class Solver(object):
    def __init__(self, args):
        super(Solver, self).__init__()
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.resume_iters = self.resume_iters
        self.is_train = args.is_train

        self.logger = args.logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build()

    def build(self):
        self.model = SiaNet(df_dim=64, ed_dim=64)
        self.model.to(self.device)
        self.init_weights(self.model, init_type="normal")
        if self.is_train:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

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

    def train(self):
        dataLoader = CasiaGait()
        lr = self.lr

        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore(self.resume_iters, is_train=self.is_train)

        for it in range(start_iters, self.num_iters):
            batch_imgs, batch_labels = dataLoader.get_batch()
            batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)
            self.optimizer.zero_grad()
            logit = self.model(batch_imgs)
            loss = ..
            loss.backward()
            self.optimizer.step()
            pass

    def restore(self, resume_iter, is_train):
        self.logger.add_log("{} Resuming model from step {}...".format(datetime.now().strftime("%Y-%m-%d %H:%M:S"), resume_iter + 1), is_print=True)
        model_path = os.path.join(self.model_dir, self.name() + '-{}.pth'.format(resume_iter))
        checkpoint = torch.load(model_path)
        if is_train:
            self.model.load_state_dict(checkpoint['id_static_dict'], strict=False)
            self.optimizer.load_state_dict(checkpoint['id_optimizer_static_dict'])
            self.model.train()
        else:
            self.model.load_state_dict(checkpoint['id_static_dict'],strict=False)
            self.model.eval()






