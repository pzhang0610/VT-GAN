import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
from utils import *
import torch
import os


def read_one_image(img_path, fine_size):
    img = Image.open(img_path).convert('L')
    img = np.asarray(img.resize(fine_size), dtype="float")
    img = transform(img)
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)
    return torch.from_numpy(img).to(torch.float32)


class CasiaGait(object):
    """Dataset class for Casia(B) dataset"""

    def __init__(self, dataset_dir, batch_size, img_size):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_size = img_size
        # self.states = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06', 'cl-01', 'cl-02', 'bg-01', 'bg-02']
        self.states = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06']
        self.angles = ['000', '018', '036', '054', '072', '090',  '108', '126', '144', '162', '180']
        self.n_id = 62
        self.attr = self.angles #+ ['nm', 'bg', 'cl']

        self.n_st = len(self.states)
        self.n_ag = len(self.angles)

    def get_batch(self):
        batch_imgs = []
        src_label = []
        dst_label = []
        spl_name = {'src_name':[], 'dst_name': [], 'neg_name':[], 'pos_name':[]}
        # src_name = []
        # dst_name = []
        # neg_name = []
        # pos_name = []

        for i in range(self.batch_size):
            # source/anchor sample
            while True:
                src_id = np.random.randint(0, self.n_id, 1).item() + 1
                src_st = np.random.randint(0, self.n_st, 1).item()
                src_ag = np.random.randint(0, self.n_ag, 1).item()
                src_spl = '%03d' % src_id + '-' + \
                    self.states[src_st] + '-' + self.angles[src_ag] + '.png'
                src_dir = os.path.join(
                    self.dataset_dir, '%03d' % src_id, self.states[src_st], src_spl)
                if os.path.exists(src_dir):
                    label = []
                    for attr in self.attr:
                        label.append(
                            attr == self.angles[src_ag])
                    src_label.append(torch.tensor(label, dtype=torch.float32))
                    spl_name['src_name'].append(src_spl)
                    break
                else:
                    # print("SRC: [{}] does not exist".format(src_spl))
                    continue
            # target sample(can be used as positive sample)
            while True:
                dst_id = src_id
                # dst_st = np.random.randint(0, self.n_st-4, 1).item()
                dst_st = np.random.randint(0, self.n_st, 1).item()
                dst_ag = np.random.randint(0, self.n_ag, 1).item()
                dst_spl = '%03d' % dst_id + '-' + \
                    self.states[dst_st] + '-' + self.angles[dst_ag] + '.png'
                dst_dir = os.path.join(
                    self.dataset_dir, '%03d' % dst_id, self.states[dst_st], dst_spl)
                if src_spl == dst_spl:
                    continue
                if os.path.exists(dst_dir):
                    label = []
                    for attr in self.attr:
                        label.append(
                            attr == self.angles[dst_ag])
                    dst_label.append(torch.tensor(label, dtype=torch.float32))
                    spl_name['dst_name'].append(dst_spl)
                    break
                else:
                    # print("DST: [{}] does not exist".format(dst_spl))
                    continue
            # negative sample
            while True:
                neg_id = np.random.randint(0, self.n_id, 1).item() + 1
                if neg_id == src_id:
                    # print("NEG: [{}] are used.".format(src_spl))
                    continue
                # neg_st = np.random.randint(0, self.n_st-4, 1).item()
                neg_st = np.random.randint(0, self.n_st, 1).item()
                neg_ag = dst_ag
                neg_spl = '%03d' % neg_id + '-' + \
                    self.states[neg_st] + '-' + self.angles[neg_ag] + '.png'
                neg_dir = os.path.join(
                    self.dataset_dir, '%03d' % neg_id, self.states[neg_st], neg_spl)
                if not os.path.exists(neg_dir):
                    # print("NEG: [{}] does not exist".format(neg_spl))
                    continue
                else:
                    spl_name['neg_name'].append(neg_spl)
                    break
            # positive sample
            while True:
                pos_id = src_id
                # pos_st = np.random.randint(0, self.n_st-4, 1).item()
                pos_st = np.random.randint(0, self.n_st, 1).item()
                pos_ag = dst_ag
                pos_spl = '%03d' % pos_id + '-' + \
                          self.states[pos_st] + '-' + self.angles[pos_ag] + '.png'
                pos_dir = os.path.join(
                    self.dataset_dir, '%03d' % pos_id, self.states[pos_st], pos_spl)
                if pos_spl == dst_spl or not os.path.exists(pos_dir):
                    # print("POS: [{}] does not exist or used".format(pos_spl))
                    continue
                else:
                    spl_name['pos_name'].append(pos_spl)
                    break

            src_img = read_one_image(src_dir, (self.img_size, self.img_size))
            dst_img = read_one_image(dst_dir, (self.img_size, self.img_size))
            neg_img = read_one_image(neg_dir, (self.img_size, self.img_size))
            pos_img = read_one_image(pos_dir, (self.img_size, self.img_size))

            image_merge = torch.cat([src_img, dst_img, neg_img, pos_img], dim=-1).permute(2, 0, 1)
            batch_imgs.append(image_merge)
        return torch.stack(batch_imgs), torch.stack(src_label), torch.stack(dst_label), spl_name

    def vis_batch(self, cols_to_display):
        images, src_label, dst_label, spl_name = self.get_batch()
        rows = int(np.ceil(self.batch_size/cols_to_display))
        for i in range(self.batch_size):
            plt.suptitle("Batch of images")
            plt.subplots_adjust(hspace=0.8, wspace=0.1, top=0.8)
            plt.subplot(rows, cols_to_display*2, 2*i+1)
            plt.axis('off')
            plt.imshow(inverse_transform(torch.squeeze(
                images[i, :1, :, :])), cmap=plt.cm.gray)
            plt.subplot(rows, cols_to_display * 2, 2 * i + 2)
            plt.axis('off')
            plt.imshow(inverse_transform(torch.squeeze(
                images[i, 1:2, :, :])), cmap=plt.cm.gray)

            # print(src_name[i], ' ', src_label[i, :],
            #       '-->', dst_name[i], ' ', dst_label[i, :], '\n')
            print('src: ', spl_name['src_name'][i], ' dst: ', spl_name['dst_name'][i], ' neg: ', spl_name['neg_name'][i], ' pos: ', spl_name['pos_name'][i])
        plt.show()


class CasiaGaitVal(object):
    """Dataset class for Casia(B) dataset"""

    def __init__(self, dataset_dir, batch_size, img_size):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_size = img_size
        # self.states = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06', 'cl-01', 'cl-02', 'bg-01', 'bg-02']
        self.states = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06']
        self.angles = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
        self.n_id = 62
        self.attr = self.angles  # + ['nm', 'bg', 'cl']

        self.n_st = len(self.states)
        self.n_ag = len(self.angles)

    def get_batch(self):
        batch_imgs = []
        src_label = []
        dst_label = []
        src_name = []
        dst_name = []

        for i in range(self.batch_size):
            # source/anchor sample
            while True:
                src_id = np.random.randint(0, self.n_id, 1).item() + 124 - self.n_id + 1
                src_st = np.random.randint(0, self.n_st, 1).item()
                src_ag = np.random.randint(0, self.n_ag, 1).item()
                src_spl = '%03d' % src_id + '-' + \
                          self.states[src_st] + '-' + self.angles[src_ag] + '.png'
                src_dir = os.path.join(
                    self.dataset_dir, '%03d' % src_id, self.states[src_st], src_spl)
                if os.path.exists(src_dir):
                    label = []
                    for attr in self.attr:
                        label.append(
                            attr == self.angles[src_ag])
                    src_label.append(torch.tensor(label, dtype=torch.float32))
                    src_name.append(src_spl)
                    break
                else:
                    continue
            # target sample(can be used as positive sample)
            while True:
                dst_id = src_id
                # dst_st = np.random.randint(0, self.n_st - 4, 1).item()
                dst_st = np.random.randint(0, self.n_st, 1).item()
                dst_ag = np.random.randint(0, self.n_ag, 1).item()
                dst_spl = '%03d' % dst_id + '-' + \
                          self.states[dst_st] + '-' + self.angles[dst_ag] + '.png'
                dst_dir = os.path.join(
                    self.dataset_dir, '%03d' % dst_id, self.states[dst_st], dst_spl)
                if src_spl == dst_spl:
                    continue
                if os.path.exists(dst_dir):
                    label = []
                    for attr in self.attr:
                        label.append(
                            attr == self.angles[dst_ag])
                    dst_label.append(torch.tensor(label, dtype=torch.float32))
                    dst_name.append(dst_spl)
                    break
                else:
                    continue
            src_img = read_one_image(src_dir, (self.img_size, self.img_size))
            dst_img = read_one_image(dst_dir, (self.img_size, self.img_size))

            image_merge = torch.cat([src_img, dst_img], dim=-1).permute(2, 0, 1)
            batch_imgs.append(image_merge)
        return torch.stack(batch_imgs), torch.stack(src_label), torch.stack(
            dst_label), src_name, dst_name


class CasiaGaitTest(object):
    def __init__(self, dataset_dir, img_size):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.states = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06', 'cl-01', 'cl-02', 'bg-01', 'bg-02']
        self.angles = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
        self.attr = self.angles  # + ['nm', 'bg', 'cl']

    def get_item(self, id, cond, angle, dst_angle):
        src_spl = '%03d' % id + '-' + \
                  self.states[cond] + '-' + self.angles[angle] + '.png'
        src_dir = os.path.join(
            self.dataset_dir, '%03d' % id, self.states[cond], src_spl)
        if not os.path.exists(src_dir):
            return -1
        img = read_one_image(src_dir, (self.img_size, self.img_size))
        label = []
        for attr in self.attr:
            label.append(attr == '%03d'%dst_angle)
        return img.permute(2, 0, 1).view(1, 1, self.img_size, self.img_size), src_spl, torch.tensor(label, dtype=torch.float32).view(1, 11)



if __name__ == "__main__":
    dataset = '../../gaitGAN/gei/'
    batch_size = 100
    dataLoader = CasiaGait(dataset, batch_size, 64)
    dataLoader.vis_batch(cols_to_display=10)
