
import argparse
import math
import time

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



def test(model, dataloader):
    average_meter = AverageMeter(dataloader.dataset.benchmark)

    all_times = []
    for idx, batch in enumerate(dataloader):
        # 1. TransforMatcher forward pass

        src_img = batch['src_img'].cuda()
        trg_img = batch['trg_img'].cuda()
        src_kps = batch['src_kps'].cuda()
        n_pts = batch['n_pts'].cuda()
        cls_id = batch['category_id']

        start_time = time.time_ns()
        corr_matrix, scale_sels = model(src_img, trg_img)

        # 2. Transfer key-points (nearest neighbor assignment)
        prd_kps = Geometry.transfer_kps_diff(corr_matrix, src_kps, n_pts, normalized=False)

        all_times.append((time.time_ns() - start_time) // 1000000)

        # 3. Evaluate predictions
        eval_result = Evaluator.evaluate(Geometry.unnormalize_kps(prd_kps), batch)
        average_meter.update(eval_result)
        average_meter.write_test_process(idx, len(dataloader))

    print("Average time per epoch:", sum(all_times) / len(all_times))

    return average_meter.get_test_result()


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='TransforMatcher Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Datasets_TransforMatcher')
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--luse', nargs='+', type=int, default=[3,4], help='layers to use (1: shallow, 4: deep)')
    parser.add_argument('--imside', type=int, default=240, help='side of an image')
    parser.add_argument('--benchmark', nargs='+', type=str, default=['pfpascal', 'pfwillow'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--bsz', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=16)
    parser.add_argument('--load', type=str, default='')

    args = parser.parse_args()

    # Model initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = transformatcher.TransforMatcher(args.backbone, args.luse, device, args.imside)

    if args.load != '': 
        model.load_state_dict(torch.load(args.load))
    else:
        print("no pretrained weights, randomly initialiZed")
        args.load = ("./logs/random_weights.log/none.pth")

    Logger.initialize(args, training=False)
    
    model.eval()

    # Test TransforMatcher for each benchmark
    for benchmark in args.benchmark:
        Logger.info('Evaluating %s...' % benchmark)
        download.download_dataset(args.datapath, benchmark)
        test_ds = download.load_dataset(benchmark, args.datapath, args.thres, False, 'test', args.imside)
        test_dl = DataLoader(test_ds, batch_size=args.bsz, shuffle=False)
        Evaluator.initialize(benchmark, device, -1)

        with torch.no_grad(): result = test(model, test_dl)
        Logger.info(result)

    Logger.info('==================== Finished Testing ====================')
