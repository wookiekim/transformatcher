
import argparse
import math

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch

from logger import Evaluator, AverageMeter, Logger
import supervision as sup
import utils as utils
from geometry import Geometry
import transformatcher as transformatcher
from data import download


def train(epoch, model, dataloader, strategy, optimizer, training):
    model.train() if training else model.eval()
    average_meter = AverageMeter(dataloader.dataset.benchmark)

    for idx, batch in enumerate(dataloader):

        # 1. TransforMatcher forward pass
        src_img, trg_img = strategy.get_image_pair(batch, training)
        corr_matrix, _ = model(src_img.cuda(), trg_img.cuda())

        # 2. Transfer key-points (weighted average)
        prd_trg_kps = Geometry.transfer_kps_diff(strategy.get_correlation(corr_matrix), batch['src_kps'].cuda(), batch['n_pts'].cuda(), normalized=False, is_train=training)

        # 3. Evaluate predictions
        eval_result = Evaluator.evaluate(Geometry.unnormalize_kps(prd_trg_kps), batch)

        # 4. Compute loss to update weights
        loss = strategy.compute_loss(corr_matrix, prd_trg_kps,
                                     batch['trg_kps'].cuda(),
                                     batch['n_pts'].cuda())
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        average_meter.update(eval_result, loss.item())
        average_meter.write_process(idx, len(dataloader), epoch)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)

    avg_loss = utils.mean(average_meter.loss_buffer)
    avg_pck = utils.mean(average_meter.buffer['pck'])
    return avg_loss, avg_pck


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='TransforMatcher Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Datasets_TransforMatcher')
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--luse', nargs='+', type=int, default=[3,4], help='layers to use (1: shallow, 4: deep)')
    parser.add_argument('--imside', type=int, default=240, help='side of an image')
    parser.add_argument('--benchmark', type=str, default='pfpascal', choices=['pfpascal', 'spair'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--finetune', type=float, default=1e-5)
    parser.add_argument('--niter', type=int, default=10)
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--nworker', type=int, default=16)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--aug', type=bool, default=True)

    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # Model initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = transformatcher.TransforMatcher(args.backbone, args.luse, device, args.imside)
    
    if args.load != '': 
        model.load_state_dict(torch.load(args.load))

    strategy = sup.StrongSupStrategy()
    optimizer = optim.Adam([{"params": model.match2match.parameters(), "lr": args.lr},
                            {"params": model.backbone.parameters(), "lr": args.finetune}])

    # Dataset download & initialization
    download.download_dataset(args.datapath, args.benchmark)
    trn_ds = download.load_dataset(args.benchmark, args.datapath, args.thres, args.aug, 'trn', args.imside)
    val_ds = download.load_dataset(args.benchmark, args.datapath, args.thres, False, 'val', args.imside)
    trn_dl = DataLoader(trn_ds, batch_size=args.bsz, num_workers=args.nworker, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.bsz, num_workers=args.nworker, shuffle=False)
    Evaluator.initialize(args.benchmark, device, args.alpha)

    # Train CHMNet
    best_val_pck = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.niter):

        trn_loss, trn_pck = train(epoch, model, trn_dl, strategy, optimizer, training=True)
        with torch.no_grad():
            val_loss, val_pck = train(epoch, model, val_dl, strategy, optimizer, training=False)

        # Save the best model
        if val_pck > best_val_pck:
            best_val_pck = val_pck
            Logger.save_model_pck(model, epoch, val_pck)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Logger.save_model_loss(model, epoch, val_loss)
        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/pck', {'trn_pck': trn_pck, 'val_pck': val_pck}, epoch)
        Logger.tbd_writer.flush()

    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
