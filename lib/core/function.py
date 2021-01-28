# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, item in enumerate(train_loader):
        if config.MODEL.USE_MASK:
            input, target, target_weight, meta, mask = item
        elif config.MODEL.USE_VECTOR:
            input, target, target_weight, meta, vector_gt = item        
        else:
            input, target, target_weight, meta = item
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        if config.MODEL.USE_VECTOR:
            output, vector_pred = model(input)
        else:
            outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if config.MODEL.USE_MASK:
            mask = mask.cuda(non_blocking=True)
            output = torch.mul(outputs, mask)
            loss = criterion(output, target, target_weight, config.MODEL.TWO_BRANCH_WEIGHT)
        elif config.MODEL.USE_VECTOR:
            vector_gt = vector_gt.cuda(non_blocking=True)
            loss_0 = criterion(output, target, target_weight, config.MODEL.TWO_BRANCH_WEIGHT)
            loss_1 = criterion(vector_pred, vector_gt, target_weight, config.MODEL.TWO_BRANCH_WEIGHT)
            loss = loss_0 + loss_1/config.MODEL.USE_VECTOR_WEIGHT
        else:
            if isinstance(outputs, list):
                loss = criterion(outputs[0], target, target_weight, config.MODEL.TWO_BRANCH_WEIGHT)
                for output in outputs[1:]:
                    loss += criterion(output, target, target_weight, config.MODEL.TWO_BRANCH_WEIGHT)
            else:
                output = outputs
                loss = criterion(output, target, target_weight, config.MODEL.TWO_BRANCH_WEIGHT)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []

    isViss = None
    if config.MODEL.BRANCH_MERGE_STRATEGY in {"all", }:
        isViss = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 1))

    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, item in enumerate(val_loader):
            if config.MODEL.USE_MASK:
                input, target, target_weight, meta, mask = item        
            elif config.MODEL.USE_VECTOR:
                input, target, target_weight, meta, vector = item        
            else:
                input, target, target_weight, meta = item
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs
            
            if config.MODEL.USE_VECTOR:
                output, vis_vector = output[0], output[1]
            
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                if config.MODEL.USE_VECTOR:
                    outputs_flipped, _ = model(input_flipped)
                else:
                    outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight, config.MODEL.TWO_BRANCH_WEIGHT)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            output_np = output.clone().cpu().numpy()
            output_vis_np = None
            if config.MODEL.USE_BRANCH:
                if config.MODEL.BRANCH_MERGE_STRATEGY == "max":
                    half_shpae = tuple(i//2 if i == config.MODEL.NUM_JOINTS*2 else i for i in output_np.shape)
                    output_np_tmp = np.zeros(half_shpae)
                    for i_output in range(output_np_tmp.shape[0]):
                        output_0 = output_np[i_output, :config.MODEL.NUM_JOINTS]
                        output_1 = output_np[i_output, config.MODEL.NUM_JOINTS:]
                        for j, (v, uv) in enumerate(zip(output_0, output_1)):
                            output_np_tmp[i_output, j] = v if np.max(v) > np.max(uv) else uv
                    output_np = output_np_tmp
                elif config.MODEL.BRANCH_MERGE_STRATEGY == "all":
                    output_np, output_vis_np = output_np[:, :config.MODEL.NUM_JOINTS], output_np[:, config.MODEL.NUM_JOINTS:]
                    # output_vis_np = None
                elif config.MODEL.BRANCH_MERGE_STRATEGY == "mix_vis_all":
                    half_shpae = tuple(i//2 if i == config.MODEL.NUM_JOINTS*2 else i for i in output_np.shape)
                    output_np_tmp = np.zeros(half_shpae)
                    for i_output in range(output_np_tmp.shape[0]):
                        output_0 = output_np[i_output, :config.MODEL.NUM_JOINTS]
                        output_1 = output_np[i_output, config.MODEL.NUM_JOINTS:]
                        for j, (v, uv) in enumerate(zip(output_0, output_1)):
                            output_np_tmp[i_output, j] = v if np.max(v) > 0.2 else uv 
                    output_np = output_np_tmp
                elif config.MODEL.BRANCH_MERGE_STRATEGY == "vis":
                    output_np = output_np[:, config.MODEL.NUM_JOINTS:]
                elif config.MODEL.BRANCH_MERGE_STRATEGY == "vis_in_all":
                    half_shpae = tuple(i//2 if i == config.MODEL.NUM_JOINTS*2 else i for i in output_np.shape)
                    output_np_tmp = np.zeros(half_shpae)
                    for i_output in range(output_np_tmp.shape[0]):
                        output_0 = output_np[i_output, :config.MODEL.NUM_JOINTS]
                        output_1 = output_np[i_output, config.MODEL.NUM_JOINTS:]
                        for j, (v_heatmap, a_heatmap) in enumerate(zip(output_0, output_1)):
                            if np.max(v_heatmap) > 0.2:
                                output_np_tmp[i_output, j] = a_heatmap
                    output_np = output_np_tmp
            if output_vis_np is not None:
                preds, maxvals, is_vis = get_final_preds(
                    config, output_np, c, s, output_vis_np)

            else:
                preds, maxvals = get_final_preds(
                    config, output_np, c, s, output_vis_np)
            if config.MODEL.USE_VECTOR:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", vis_vector.shape)
                vis_vector = vis_vector.cpu().numpy().reshape((num_images, config.MODEL.NUM_JOINTS, 1))
                for index_0 in range(vis_vector.shape[0]):
                    for index_1 in range(vis_vector.shape[1]):
                        for index_2 in range(vis_vector.shape[2]):
                            old_v = vis_vector[index_0, index_1, index_2]
                            if old_v > 0.2:
                                vis_vector[index_0, index_1, index_2] = 2
                            else:
                                vis_vector[index_0, index_1, index_2] = 1
                isViss[idx:idx + num_images, :] = vis_vector
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            if output_vis_np is not None:
                isViss[idx:idx + num_images, :] = is_vis
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums, is_vis=isViss
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
