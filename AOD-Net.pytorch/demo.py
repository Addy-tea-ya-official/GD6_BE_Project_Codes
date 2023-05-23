import os
import glob
import torch
import torch.backends.cudnn
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from PIL import Image
from utils import logger
from config import get_config
from model import AODnet

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = (os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

@logger
def make_test_data(cfg, img_path_list, device):
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((480, 720)),
        torchvision.transforms.ToTensor()
    ])
    imgs = []
    for img_path in img_path_list:
        x = data_transform(Image.open(img_path)).unsqueeze(0)
        x = x.to(device)
        imgs.append(x)
    return imgs


@logger
def load_pretrain_network(cfg, device):
    net = AODnet().to(device)
    net.load_state_dict(torch.load('D:/dataset/AOD-Net.pytorch/model/nets/AOD_19.pkl')['state_dict'])
    return net


def main(cfg):
    # -------------------------------------------------------------------
    # basic config
    imgs=load_images_from_folder("D:\\dataset\\MIO-TCD-Localization\\MIO-TCD-Localization\\_480x720")
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------------
    # load data
    #test_file_path = glob.glob('D:dataset/part3/*.jpg')
    #test_images = make_test_data(cfg, test_file_path, device)
    # -------------------------------------------------------------------
    # load network
    network = load_pretrain_network(cfg, device)
    # -------------------------------------------------------------------
    # set network weights
    # -------------------------------------------------------------------
    # start train
    print('Start eval')
    network.eval()
    for i in imgs:
        test_file_path = glob.glob(i)
        test_images = make_test_data(cfg, test_file_path, device)
        for idx, im in enumerate(test_images):
            dehaze_image = network(im)
            print(dehaze_image)
            print(test_file_path[idx])
            torchvision.utils.save_image(dehaze_image, "MIO-TCD_aod_720_480\\" + test_file_path[idx].split("\\")[-1])


if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)
