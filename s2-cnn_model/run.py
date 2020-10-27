import json
import os
import pickle

import fire
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as tdata
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch.nn.functional as F

from dataset import MyDataset, get_loader_pair, prepare_folds
from model import get_model
from xz import debug, display_imgs
from ztorch.engine import Engine
from ztorch.plugins import Checkpoint, TrainingMetrics, Messages, ReduceLROnPlateau, Timestamp, ValidationMetrics

# 20190609
# resnet18:
#     AUC = <around 0.52>
#     -----------------------
#     base model = resnet18
#     20% split validation
#     resolution = 384x384
#     batch_size = 24
#     max_epochs = 100

# 20190611
# resnet18_single
#     AUC = 0.55
#     -----------------------
#     from <resnet18>
#     changed from two semis to one single resnet18

# 20190611
# v3
#     AUC = 0.55
#     -----------------------
#     from <resnet18_single>
#     bumped resolution to 512

# 20190613
# v3_pres
#     AUC = 0.55
#     -----------------------
#     from <v3>
#     used processed proc2pres outputs

# 20190613
# v3_densenet121
#     AUC = ??
#     -----------------------
#     from <v3>
#     densenet121

# 20190614
# v3_densenet121_pres
#     AUC = ??
#     -----------------------
#     from <v3_densenet121>
#     used processed proc2pres outputs

# 20190616
# v3_densenet121_pres_pretrained
#     AUC = ??
#     -----------------------
#     from <v3_densenet121_pres>
#     imagenet pretrained

# 20190617
# v3_densenet121_pres_pretrained_2
#     AUC = ??
#     -----------------------
#     from <v3_densenet121_pres_pretrained>
#     another independent try

# 20190625
# v3_gerasnet_pres_pretrained
#     AUC = ??
#     -----------------------
#     from <v3_densenet121_pres_pretrained>
#     changed model to gerasnet

# 20190627
# v3_gerasnet_pres_pretrained_last_layer_frozen
#     AUC = ??
#     -----------------------
#     from <v3_gerasnet_pres_pretrained>
#     froze last layer

# 20190629
# v3_gerasnet_raw_pretrained
#     AUC = ??
#     -----------------------
#     from <v3_gerasnet_pres_pretrained>
#     changed from pres to raw

# 20190630
# v3_gerasnet_raw_pretrained_last_layer_frozen
#     AUC = ??
#     -----------------------
#     from <v3_gerasnet_raw_pretrained>
#     froze last layer

# 20190707
# v4_densenet121_hl512_imagenet_train_directly_with_3_classes
#     AUC = ??
#     -----------------------
#     from <v3_densenet121>
#     used a low/high freq two-channel input
#     imagenet-pretrained
#     trained directly with three classes

# 20190708
# v4_densenet121_hl512_imagenet
#     AUC = ??
#     -----------------------
#     from <v3_densenet121>
#     used a low/high freq two-channel input
#     imagenet-pretrained

config = {
    'n_classes': 1,
    'device': 'cuda:4',
    'train_batch_size': 4,
    'pct_valid': 0.2,
    'dir_to_npys': 'data/hl_512_640/npys_for_cnn',
    'path_to_pmeta': 'data/hl_512_640/pmeta.csv',
    'pwd': 'tmp/test',
    'single_channel': False,
    'phase': 'p1',
    'max_epochs': 100,
    'n_folds': 5,
}


def preview_batches(config=config):
    pmeta = pd.read_csv(config['path_to_pmeta'])
    pmeta = pmeta.loc[pmeta['group'] == 'train']
    dataset = MyDataset(pmeta, config, debug=True)

    data_loader = tdata.DataLoader(
        dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        num_workers=70,
        collate_fn=dataset.collate_fn,
    )

    img_list = next(iter(data_loader))
    display_imgs(img_list, save_as='tmp/batch_preview.png')


def predict_test(path_to_checkpoint, group='test', truth=True):
    with open(path_to_checkpoint, 'rb') as f:
        engine = pickle.load(f)

    # dataset = engine.valid_loader.dataset
    # pmeta = dataset.pmeta.copy()

    config = {
        'device': 'cuda:6',
        'train_batch_size': 4,
        'dir_to_npys': 'data/pres512_2/npys_for_cnn',
        'path_to_pmeta': 'data/pres512_2/pmeta.csv',
        'max_epochs': 100,
        'single_channel': True,
        'phase': 'p2',
    }

    pmeta = pd.read_csv(config['path_to_pmeta'])
    pmeta = pmeta.loc[pmeta['group'] == group]

    dataset = MyDataset(pmeta, config, debug=False)

    valid_loader = tdata.DataLoader(
        dataset,
        batch_size=config['train_batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn,
    )

    yp_list = []
    if truth:
        y_list = []
    engine.model.to(config['device'])
    engine.model.eval()
    with torch.no_grad():
        for i_iter, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            x, y = engine.process_batch_fn(batch, config['device'])
            print(f"x =\n{x}")
            print(f"x.size() =\n{x.size()}")
            print(f"engine.model = {engine.model}")
            yp = engine.model(x)
            if truth:
                y_list.append(y)
            yp_list.append(yp)

    if truth:
        ys = torch.cat(y_list)
        ys = ys.cpu().numpy()

    yps = torch.cat(yp_list)
    yps = yps.cpu().numpy()

    torch.save(yps, 'tmp/yps.pth')

    yps = torch.load('tmp/yps.pth')

    pmeta['control'] = yps[:, 0]
    pmeta['screen_detected'] = yps[:, 1]
    pmeta['interval'] = yps[:, 2]
    # pmeta['yp'] = np.argmax(yps, axis=1)
    # if truth:
    #     pmeta['y'] = np.argmax(ys, axis=1)

    if truth:
        print(pmeta)
        print(f"control: {roc_auc_score(pmeta['y'] == 0, pmeta['control'])}")
        print(f"screen_detected: {roc_auc_score(pmeta['y'] == 1, pmeta['screen_detected'])}")
        print(f"interval: {roc_auc_score(pmeta['y'] == 2, pmeta['interval'])}")

    torch.save(pmeta, 'tmp/prediction_df.pth')
    pmeta.to_csv('tmp/prediction_df.csv', index=False)


def resume(path_to_checkpoint, pwd=None):
    with open(path_to_checkpoint, 'rb') as f:
        engine = pickle.load(f)

    engine.i_epoch += 1

    engine.max_epochs += 100
    old = engine.max_epochs
    engine.max_epochs += 100
    debug(f"engine.max_epochs is increased from {old} to {engine.max_epochs}")

    if pwd is not None:
        old = engine.pwd
        engine.pwd = pwd
        debug(f"engine.pwd is changed from {old} to {engine.pwd}")

    engine.run()


# preview_batches(config)

# history = pd.read_csv('tmp/find_lr.csv', index_col=0)
# history = history.iloc[:-13]
# plt.figure(figsize=(10, 5))
# plt.plot(history["lr"], history["running_loss"])
# plt.xscale("log")
# plt.xlabel("Learning rate")
# plt.ylabel("Loss")
# plt.savefig('tmp/find_lr.pdf')

# def parse_messages(path_to_messages):
#     rows = []

#     with open(path_to_messages, 'r') as f:
#         for line in f:
#             cells = line.rstrip('\n').split(' | ')
#             row = {}
#             for cell in cells:
#                 k, *v = cell.split(' ')
#                 row[k] = ' '.join(v)
#             rows.append(row)

#     msg_df = pd.DataFrame.from_records(rows)
#     return msg_df


def parse_messages(path_to_messages):
    rows = []

    with open(path_to_messages, 'r') as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)

    msg_df = pd.DataFrame.from_records(rows)
    return msg_df.iloc[:100]


def plot():
    # toy_msg_df = parse_messages('working/messages.txt')
    # toy_msg_df.to_csv('tmp/toy.csv')

    # list_of_pwds = ['v3', 'v3_densenet121', 'v3_densenet121_pres', 'v3_densenet121_pres_pretrained']
    list_of_pwds = [
        'v3_densenet121_pres_pretrained',
        'v3_gerasnet_pres_pretrained',
        'v3_gerasnet_pres_pretrained_last_layer_frozen',
        'v3_gerasnet_raw_pretrained',
        'v3_gerasnet_raw_pretrained_last_layer_frozen',
    ]
    dfs = {k: parse_messages(f"working/{k}/log.json") for k in list_of_pwds}
    colors = ['red', 'blue', 'green', 'lightgreen', 'gray']

    plt.figure(figsize=(10, 5))
    for i, (k, df) in enumerate(dfs.items()):
        # plt.plot(
        #     df.index,
        #     df['avg_train_loss'].astype(np.float32),
        #     color='red',
        #     label='[BRCA] Avg. training loss',
        # )
        plt.plot(
            df.index,
            df['val_auc'].astype(np.float32),
            # linestyle=linestyles[i],
            color=colors[i],
            label=k,
        )
    plt.ylim(0.50, 0.65)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.savefig('tmp/tmp.png', dpi=100)


def tmp():
    config = {
        'n_classes': 1,
        'device': 'cuda:5',
        'train_batch_size': 4,
        'pct_valid': 0.2,
        'dir_to_npys': 'data/pres512/npys_for_cnn',
        'path_to_pmeta': 'data/pres512/pmeta.csv',
        'pwd': 'working/v3_densenet121_pres_pretrained__3_classes',
        'max_epochs': 100,
    }

    train_pmeta = pd.read_csv('working/v3_densenet121_pres_pretrained/train_pmeta.csv', index_col=0)
    train_dataset = MyDataset(train_pmeta, config, debug=False)
    train_loader = tdata.DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        num_workers=70,
        collate_fn=train_dataset.collate_fn,
    )

    valid_pmeta = pd.read_csv('working/v3_densenet121_pres_pretrained/valid_pmeta.csv', index_col=0)
    valid_dataset = MyDataset(valid_pmeta, config, debug=False)
    valid_loader = tdata.DataLoader(
        valid_dataset,
        batch_size=config['train_batch_size'],
        shuffle=False,
        num_workers=70,
        collate_fn=valid_dataset.collate_fn,
    )

    # MODEL BEGIN -------------------
    model = get_model(config)
    sds = torch.load('pretrained_weights/v3_densenet121_pres_pretrained_e4.pth')
    model.load_state_dict(sds['model_state_dict'])

    # freeze all but the last layer
    for param in model.basemodel_cc.parameters():
        param.requires_grad = False

    model.classifier = nn.Linear(1024, 3)
    # MODEL END -------------------

    engine = Engine(
        model=nn.Sequential(model, nn.Softmax()),
        optimizer=(torch.optim.Adam, {'lr': 3e-4}),
        loss_fn=nn.BCELoss(),
        # loss_fn=nn.BCEWithLogitsLoss(),
        max_epochs=config['max_epochs'],
        loader_pair=(train_loader, valid_loader),
        plugins=[
            Timestamp(),
            TrainingMetrics(),
            ValidationMetrics(),
            ReduceLROnPlateau(),
            Checkpoint(),
            Messages(),
        ],
        device=config['device'],
        pwd=config['pwd'],
    )

    # engine.optimizer.load_state_dict(sds['optimizer_state_dict'])

    engine.run()


def loss(yp, y):
    return F.binary_cross_entropy_with_logits(yp, y).item()


def acc(yp, y):
    yps = torch.sigmoid(yp)
    ypsr = torch.round(yps)

    # TODO: DEBUG
    torch.save(y, 'tmp/tmp.pth')
    # mat = torch.cat([y, ypsr, yps, yp], 1).cpu().numpy()
    # print(f"mat = {mat}")
    # pred_df = pd.DataFrame(mat, columns=['y', 'ypsr', 'yps', 'yp'])
    # pred_df.to_csv('tmp/tmp.csv')
    # TODO: DEBUG END

    jud = 1 - (y.type(torch.uint8) ^ ypsr.type(torch.uint8))
    acc = torch.mean(jud.type(torch.float)).item()

    return acc
    # return {
    #     'accuracy': acc,
    #     'correct': torch.sum(jud).item(),
    #     'n_total': len(jud),
    # }


def std(yp, y):
    return {
        'yps_std': torch.sigmoid(yp).std(dim=0).item(),
        'y_std': y.std(dim=0).item(),
    }


def auc(yp, y):
    return roc_auc_score(y.cpu().numpy(), yp.cpu().numpy())


def timestamp(yp, y):
    return datetime.now()


def train_folds(do_test_run=False, n_folds=5):
    print(f"config =\n{config}")

    if os.path.exists(config['pwd']):
        if input(f"{config['pwd']} already exists, are you sure? [y/N]: ") != 'y':
            return

    prepare_folds(config)



def main(do_test_run=False, **kwargs):
    print(f"config =\n{config}")

    config.update(kwargs)

    if os.path.exists(config['pwd']):
        if input(f"{config['pwd']} already exists, are you sure? [y/N]: ") != 'y':
            return

    engine = Engine(
        model=get_model(config),
        optimizer=(torch.optim.Adam, {'lr': 3e-4}),
        # loss_fn=nn.BCELoss(),
        loss_fn=nn.BCEWithLogitsLoss(),
        max_epochs=config['max_epochs'],
        loader_pair=get_loader_pair(config),
        plugins=[
            Timestamp(),
            TrainingMetrics({
                'loss': loss,
                'acc': acc,
            }, residual_factor=max(1 - config['train_batch_size'] / 10000, 0)),
            ValidationMetrics({
                'loss': loss,
                'acc': acc,
                'std': std,
                'auc': auc,
            }),
            ReduceLROnPlateau(),
            Checkpoint(),
            Messages(),
        ],
        device=config['device'],
        pwd=config['pwd'],
    )

    if do_test_run:
        engine.run(test_run=True)
        engine.reset()
    engine.run()


def view_history(
    logs,
    cols=None,
    save_as=None,
):
    df = pd.DataFrame.from_records(logs)

    if cols is not None:
        df = df[cols]

    if save_as is None:
        print(df)
    else:
        df.to_csv(save_as)
        debug(f"Saved as {save_as}")


def plot_history(
    logs,
    cols=['avg_train_loss', 'val_loss'],
    save_as=None,
    figsize=(10, 5),
    dpi=100,
):
    df = pd.DataFrame.from_records(logs)

    plt.figure(figsize=figsize)

    for col in cols:
        plt.plot(
            df.index,
            df[col].astype(np.float),
            label=col,
        )

    plt.ylim(0, 1)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if save_as is None:
        plt.show()
    else:
        plt.savefig(save_as, dpi=dpi)
        debug(f"Saved as {save_as}")


if __name__ == '__main__':
    fire.Fire()
