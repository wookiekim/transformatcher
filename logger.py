import datetime
import logging
import os
import shutil
import numpy as np

from tensorboardX import SummaryWriter
import torch

import utils

class Logger:
    r"""Writes results of training/testing"""
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath if training else '_TEST_' + args.load.split('/')[-2].split('.')[0] + logtime
        if logpath == '': logpath = logtime

        cls.logpath = os.path.join('logs', logpath + '.log')
        cls.benchmark = args.benchmark
        if os.path.isdir(cls.logpath):
            if logpath == 'debug':
                shutil.rmtree(cls.logpath)
            else:
                resp = input("Existing folder. Overwrite? Y/N: ")

                if resp in ['Y','y']:
                    shutil.rmtree(cls.logpath)
        os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        if training:
            logging.info('\n:======== TransforMatcher ========')
            for arg_key in args.__dict__:
                logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
            logging.info(':=================================\n')


    @classmethod
    def info(cls, msg):
        r"""Writes message to .txt"""
        logging.info(msg)

    @classmethod
    def save_model_pck(cls, model, epoch, val_pck):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'pck_best_model.pt'))
        cls.info('Model saved @%d w/ val. PCK: %5.2f.\n' % (epoch, val_pck))

    @classmethod
    def save_model_loss(cls, model, epoch, val_loss):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'loss_best_model.pt'))
        cls.info('Model saved @%d w/ val. loss: %5.2f.\n' % (epoch, val_loss))

    @classmethod
    def save_model(cls, model, epoch):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'last_model.pt'))
        cls.info('Model saved @%d.\n' % (epoch))

class AverageMeter:
    r"""Stores loss, evaluation results, selected layers"""
    def __init__(self, benchamrk):
        r"""Constructor of AverageMeter"""
        if benchamrk == 'caltech':
            self.buffer_keys = ['ltacc', 'iou']
        else:
            self.buffer_keys = ['pck']

        self.buffer = {}
        for key in self.buffer_keys:
            self.buffer[key] = []

        self.loss_buffer = []

    def update(self, eval_result, loss=None):
        for key in self.buffer_keys:
            self.buffer[key] += eval_result[key]

        if loss is not None:
            self.loss_buffer.append(loss)

    def write_result(self, split, epoch):
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch

        if len(self.loss_buffer) > 0:
            msg += 'Loss: %5.2f  ' % (sum(self.loss_buffer) / len(self.loss_buffer))

        for key in self.buffer_keys:
            msg += '%s: %6.2f  ' % (key.upper(), sum(self.buffer[key]) / len(self.buffer[key]))
        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch):
        msg = '[Epoch: %02d] ' % epoch
        msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
        if len(self.loss_buffer) > 0:
            msg += 'Loss: %6.2f  ' % self.loss_buffer[-1]
            msg += 'Avg Loss: %6.5f  ' % (sum(self.loss_buffer) / len(self.loss_buffer))

        for key in self.buffer_keys:
            msg += 'Avg %s: %6.2f  ' % (key.upper(), sum(self.buffer[key]) / len(self.buffer[key]))
        Logger.info(msg)

    def write_test_process(self, batch_idx, datalen):
        msg = '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)

        for key in self.buffer_keys:
            if key == 'pck':
                pcks = torch.stack(self.buffer[key]).mean(dim=0) * 100
                val = ''
                for p in pcks:
                    val += '%5.2f   ' % p.item()
                msg += 'Avg %s: %s   ' % (key.upper(), val)
            else:
                msg += 'Avg %s: %6.2f  ' % (key.upper(), sum(self.buffer[key]) / len(self.buffer[key]))
        Logger.info(msg)

    def get_test_result(self):
        result = {}
        for key in self.buffer_keys:
            if key == 'pck':
                result[key] = torch.stack(self.buffer[key]).mean(dim=0) * 100
            else:
                result[key] = sum(self.buffer[key]) / len(self.buffer[key])

        return result

class Evaluator:
    r"""Computes evaluation metrics of PCK, LT-ACC, IoU"""
    @classmethod
    def initialize(cls, benchmark, device, alpha=0.1):
        if alpha == -1:
            cls.eval_func = cls.eval_kps_transfer_test
            cls.alpha = torch.tensor([0.05, 0.1, 0.15]).unsqueeze(1).to(device)
        else:
            cls.eval_func = cls.eval_kps_transfer
            cls.alpha = alpha

    @classmethod
    def evaluate(cls, prd_kps, batch):
        r"""Compute evaluation metric"""
        return cls.eval_func(prd_kps, batch)

    @classmethod
    def classify_prd(cls, prd_kps, trg_kps, pckthres):
        r"""Compute the number of correctly transferred key-points"""
        l2dist = (prd_kps - trg_kps).pow(2).sum(dim=0).pow(0.5)
        thres = pckthres.expand_as(l2dist).float() * cls.alpha
        correct_pts = torch.le(l2dist, thres)

        correct_ids = utils.where(correct_pts == 1)
        incorrect_ids = utils.where(correct_pts == 0)
        correct_dist = l2dist[correct_pts]

        return correct_dist, correct_ids, incorrect_ids    

    @classmethod
    def eval_kps_transfer(cls, prd_kps, batch):
        r"""Compute percentage of correct key-points (PCK)"""

        pck = []
        for idx, (pk, tk) in enumerate(zip(prd_kps, batch['trg_kps'])):
            thres = batch['pckthres'][idx].cuda()
            npt = batch['n_pts'][idx]
            correct_dist, correct_ids, incorrect_ids = cls.classify_prd(pk[:, :npt].cuda(),
                                                                        tk[:, :npt].cuda(),
                                                                        thres)
            pck.append((len(correct_ids) / npt.item()) * 100)

        eval_result = {'pck': pck}

        return eval_result

    @classmethod
    def eval_kps_transfer_test(cls, prd_kps, batch):
        r"""Compute percentage of correct key-points (PCK) with multiple alpha {0.05, 0.1, 0.15}"""

        pck = []
        for idx, (pk, tk) in enumerate(zip(prd_kps, batch['trg_kps'])):
            pckthres = batch['pckthres'][idx].cuda()
            npt = batch['n_pts'][idx]
            prd_kps = pk[:, :npt].cuda()
            trg_kps = tk[:, :npt].cuda()

            l2dist = (prd_kps - trg_kps).pow(2).sum(dim=0).pow(0.5).unsqueeze(0).repeat(len(cls.alpha), 1)
            thres = pckthres.expand_as(l2dist).float() * cls.alpha
            pck.append(torch.le(l2dist, thres).sum(dim=1) / float(npt))

        eval_result = {'pck': pck}

        return eval_result