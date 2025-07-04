import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from source.utils import load_model
from source.models.model_factory import Model
from main import Config
from source.dataset import SplitDataset, get_data
import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import time
from sklearn.metrics import confusion_matrix


configs = Config()
configs.path_root = '../dataset/timeseries/dataset/20200603'
configs.model_name = 'predrnn'
# model definition
model = Model(configs)
model = load_model(model, 'checkpoints/predrnn/model-95.ckpt')
model = model.to(configs.device)
data = get_data(configs)
dataset = SplitDataset(data, configs)
len = dataset.__len__()
testdl = DataLoader(dataset, batch_size=configs.batch_size)
avg_mse = 0
ssim_score = []
confma = np.zeros((2,2))
with torch.no_grad():
    model.eval()
    test_losses = []
    start = time.time()
    for i, batch in enumerate(testdl):
        x, mask = batch
        x = x.to(configs.device)
        mask = mask.to(configs.device)
        pred = model.forward(x, mask)
        loss = F.mse_loss(pred, x[:, 1:])
        x = x.to('cpu')
        pred = pred.to('cpu')
        x_flat = torch.flatten(x[:, 1:]).numpy()
        print(np.mean(x_flat))
        x_flat = np.where(x_flat > 0.12, 1, 0)
        pred_flat = torch.flatten(pred).numpy()
        print(np.mean(pred_flat))
        pred_flat = np.where(pred_flat > 0.12, 1, 0)
        a = confusion_matrix(x_flat, pred_flat)
        confma += a
        mse = np.square(pred - x[:, 1:]).sum()
        avg_mse += mse
        test_losses.append(loss.item())
        real_frm = np.uint8(x[:, 1:] * 255)
        pred_frm = np.uint8(pred * 255)

        for b in range(configs.batch_size):
            s, _ = ssim(pred_frm[b], real_frm[b], full=True, multichannel=True)
            ssim_score.append(s)



    end = time.time()
    print(end-start)
    csi = confma[0]/(confma[0][0] + confma[0][1] + confma[1][0])
    print(csi)
    print(configs.model_name + f' csi: ', csi.astype(str))
    print(configs.model_name + f' test loss: {np.mean(test_losses):.5f}')
    print(configs.model_name + f' mse: {mse/len:.5f}')
    print(configs.model_name + f' ssim: {np.average(ssim_score):.5f}')