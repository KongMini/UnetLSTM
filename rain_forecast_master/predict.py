import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from source.utils import load_model
from source.models.model_factory import Model
from main import Config
from source.dataset import SplitDataset, get_data
from mpl_toolkits.axes_grid1 import make_axes_locatable


def show_image(dataset, configs, index=1):
    if os.path.exists('result_images') is False:
        os.makedirs('result_images')
    model_names = ['input', 'groundtruth', 'predrnn', 'predrnn_v2', 'unet_predrnn_v2']
    for name in model_names:
        if name == 'input':
            fig, a = plt.subplots(2, 3, constrained_layout=True, figsize=(12, 8))
            sample, mask = dataset[index]
            sample = sample.unsqueeze(0).to(configs.device)
            for i in range(configs.pred_length):
                if i <= 2:
                    im = a[0][i].imshow(sample[0, i + 1, :, :, :].cpu().numpy(), cmap='gray')
                    a[0][i].axis('off')
                    a[0][i].set_title('t - ' + str(5 - i), fontsize=24)
                else:
                    im = a[1][i - 3].imshow(sample[0, i + 1, :, :, :].cpu().numpy(), cmap='gray')
                    a[1][i - 3].axis('off')
                    if i == 5:
                        a[1][i - 3].set_title('t', fontsize=24)
                    else:
                        a[1][i - 3].set_title('t - ' + str(5 - i), fontsize=24)

            cb = fig.colorbar(im, ax=a.ravel().tolist(), shrink=0.75)
            sample, mask = dataset[index]
            sample = sample.unsqueeze(0).to(configs.device)
            for i in range(configs.pred_length):
                if i <= 2:
                    im = a[0][i].imshow(sample[0, i + 7, :, :, :].cpu().numpy(), cmap='gray')
                    a[0][i].axis('off')
                    a[0][i].set_title('t + ' + str(i + 1), fontsize=24)
                else:
                    im = a[1][i - 3].imshow(sample[0, i + 7, :, :, :].cpu().numpy(), cmap='gray')
                    a[1][i - 3].axis('off')
                    a[1][i - 3].set_title('t + ' + str(i + 1), fontsize=24)
            cb = fig.colorbar(im, ax=a.ravel().tolist(), shrink=0.75)
            cb.set_label(label='rain level', size=24)
            fig.savefig('result_images/groundtruth.png')
        else:
            fig, a = plt.subplots(2, 3, constrained_layout=True, figsize=(12, 8))
            configs.model_name = name
            model = Model(configs)
            if name == 'predrnn':
                model = load_model(model, 'checkpoints/'+ name +'/model-30.ckpt')
            else:
                model = load_model(model, 'checkpoints/'+ name +'/model-95.ckpt')
            model = model.to(configs.device)
            with torch.no_grad():
                sample, mask = dataset[index]
                sample = sample.unsqueeze(0).to(configs.device)
                mask = mask.unsqueeze(0).to(configs.device)
                model.to(configs.device)
                pred = model(sample, mask)
                for i in range(configs.pred_length):
                    if i <= 2:
                        im = a[0][i].imshow(pred[0, i + 6, :, :, :].cpu().numpy()*255, cmap='gray')
                        a[0][i].axis('off')
                        a[0][i].set_title('t + ' + str(i + 1), fontsize=24)
                    else:
                        im = a[1][i - 3].imshow(pred[0, i + 6, :, :, :].cpu().numpy()*255, cmap='gray')
                        a[1][i - 3].axis('off')
                        a[1][i - 3].set_title('t + ' + str(i + 1), fontsize=24)
                cb = fig.colorbar(im, ax=a.ravel().tolist(), shrink=0.75)
                cb.set_label(label='rain level', size=24)
            fig.savefig('result_images/'+name+'.png')


if __name__ == '__main__':
    configs = Config()
    configs.total_length = 13
    configs.input_length = 7
    configs.batch_size = 1
    configs.pred_length = 6
    data = get_data(configs)
    dataset = SplitDataset(data, configs)
    show_image(dataset, configs, 60)
    print('Done!')
