# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

import timm
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.cosub:
            samples = torch.cat((samples, samples), dim=0)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        with amp_autocast():
            outputs = model(samples, if_random_cls_token_position=args.if_random_cls_token_position,
                            if_random_token_rank=args.if_random_token_rank)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0] // 2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets)
                loss = loss + 0.25 * criterion(outputs[1], targets)
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid())

        if args.if_nan2num:
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)

        optimizer.zero_grad()

        # ============== 初始化梯度监控变量 ==============
        grad_norms = {}  # 确保这个变量在所有分支之前定义
        grad_stats = {
            'total_norm': 0.0,
            'max_norm': 0.0,
            'min_norm': float('inf'),
            'max_layer': None,
            'min_layer': None,
            'nan_count': 0,
            'inf_count': 0
        }
        # ===========================================

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)

            # 计算梯度
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        grad_norms[name] = grad_norm
                        grad_stats['total_norm'] += grad_norm ** 2

                        if torch.isnan(param.grad).any():
                            grad_stats['nan_count'] += 1
                        if torch.isinf(param.grad).any():
                            grad_stats['inf_count'] += 1

                        if grad_norm > grad_stats['max_norm']:
                            grad_stats['max_norm'] = grad_norm
                            grad_stats['max_layer'] = name

                        if grad_norm < grad_stats['min_norm']:
                            grad_stats['min_norm'] = grad_norm
                            grad_stats['min_layer'] = name

                grad_stats['total_norm'] = grad_stats['total_norm'] ** 0.5
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # 计算梯度
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        grad_norms[name] = grad_norm
                        grad_stats['total_norm'] += grad_norm ** 2

                        if torch.isnan(param.grad).any():
                            grad_stats['nan_count'] += 1
                        if torch.isinf(param.grad).any():
                            grad_stats['inf_count'] += 1

                        if grad_norm > grad_stats['max_norm']:
                            grad_stats['max_norm'] = grad_norm
                            grad_stats['max_layer'] = name

                        if grad_norm < grad_stats['min_norm']:
                            grad_stats['min_norm'] = grad_norm
                            grad_stats['min_layer'] = name

                grad_stats['total_norm'] = grad_stats['total_norm'] ** 0.5

            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # ============== 梯度问题检测和日志记录 ==============
        # 记录到metric_logger
        metric_logger.update(grad_norm=grad_stats['total_norm'])
        metric_logger.update(max_grad_norm=grad_stats['max_norm'])
        metric_logger.update(min_grad_norm=grad_stats['min_norm'])

        # 梯度问题检测
        if grad_stats['total_norm'] > 1000:  # 梯度爆炸阈值
            print(f"⚠️ Gradient explosion detected! Total norm: {grad_stats['total_norm']:.2f}")
            # 记录最大梯度层
            if args.local_rank == 0 and args.gpu == 0:
                mlflow.log_metric("grad_explosion/max_layer", grad_stats['max_layer'], step=metric_logger.step)

        if grad_stats['total_norm'] < 1e-5:  # 梯度消失阈值
            print(f"⚠️ Gradient vanishing detected! Total norm: {grad_stats['total_norm']:.2e}")

        if grad_stats['nan_count'] > 0:
            print(f"⚠️ NaN gradients detected in {grad_stats['nan_count']} layers!")

        if grad_stats['inf_count'] > 0:
            print(f"⚠️ Inf gradients detected in {grad_stats['inf_count']} layers!")

        # MLflow日志记录
        if args.local_rank == 0 and args.gpu == 0:
            mlflow.log_metric("grad_norm/total", grad_stats['total_norm'], step=metric_logger.step)
            mlflow.log_metric("grad_norm/max", grad_stats['max_norm'], step=metric_logger.step)
            mlflow.log_metric("grad_norm/min", grad_stats['min_norm'], step=metric_logger.step)
            mlflow.log_metric("grad_norm/nan_count", grad_stats['nan_count'], step=metric_logger.step)
            mlflow.log_metric("grad_norm/inf_count", grad_stats['inf_count'], step=metric_logger.step)

            # 每100步记录一次详细梯度分布
            if metric_logger.step % 100 == 0:
                for name, norm in grad_norms.items():
                    # 简化层名，避免MLflow中的命名过长问题
                    simplified_name = name.replace('.', '_')[:50]
                    mlflow.log_metric(f"grad_norms/{simplified_name}", norm, step=metric_logger.step)
        # ===========================================

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
#                     set_training_mode=True, args = None):
#     model.train(set_training_mode)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 10
#
#     if args.cosub:
#         criterion = torch.nn.BCEWithLogitsLoss()
#
#     # debug
#     # count = 0
#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
#         # count += 1
#         # if count > 20:
#         #     break
#
#         samples = samples.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)
#
#         if mixup_fn is not None:
#             samples, targets = mixup_fn(samples, targets)
#
#         if args.cosub:
#             samples = torch.cat((samples,samples),dim=0)
#
#         if args.bce_loss:
#             targets = targets.gt(0.0).type(targets.dtype)
#
#         with amp_autocast():
#             outputs = model(samples, if_random_cls_token_position=args.if_random_cls_token_position, if_random_token_rank=args.if_random_token_rank)
#             # outputs = model(samples)
#             if not args.cosub:
#                 loss = criterion(samples, outputs, targets)
#             else:
#                 outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
#                 loss = 0.25 * criterion(outputs[0], targets)
#                 loss = loss + 0.25 * criterion(outputs[1], targets)
#                 loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
#                 loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid())
#
#         if args.if_nan2num:
#             with amp_autocast():
#                 loss = torch.nan_to_num(loss)
#
#         loss_value = loss.item()
#
#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             if args.if_continue_inf:
#                 optimizer.zero_grad()
#                 # ============== 梯度监控变量 ==============
#                 grad_norms = {}
#                 grad_stats = {
#                     'total_norm': 0.0,
#                     'max_norm': 0.0,
#                     'min_norm': float('inf'),
#                     'max_layer': None,
#                     'min_layer': None,
#                     'nan_count': 0,
#                     'inf_count': 0
#                 }
#                 # ========================================
#                 continue
#             else:
#                 sys.exit(1)
#
#         optimizer.zero_grad()
#
#         # this attribute is added by timm on one optimizer (adahessian)
#         if isinstance(loss_scaler, timm.utils.NativeScaler):
#             is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#             loss_scaler(loss, optimizer, clip_grad=max_norm,
#                     parameters=model.parameters(), create_graph=is_second_order)
#
#             # 添加梯度计算代码（NativeScaler分支）
#             with torch.no_grad():
#                 for name, param in model.named_parameters():
#                     if param.grad is not None:
#                         grad_norm = param.grad.data.norm(2).item()
#                         grad_norms[name] = grad_norm
#                         grad_stats['total_norm'] += grad_norm ** 2
#
#                         if torch.isnan(param.grad).any():
#                             grad_stats['nan_count'] += 1
#                         if torch.isinf(param.grad).any():
#                             grad_stats['inf_count'] += 1
#
#                         if grad_norm > grad_stats['max_norm']:
#                             grad_stats['max_norm'] = grad_norm
#                             grad_stats['max_layer'] = name
#
#                         if grad_norm < grad_stats['min_norm']:
#                             grad_stats['min_norm'] = grad_norm
#                             grad_stats['min_layer'] = name
#
#                 grad_stats['total_norm'] = grad_stats['total_norm'] ** 0.5
#
#         else:
#             loss.backward()
#             if max_norm != None:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#
#                 # 添加梯度计算代码（标准分支）
#             with torch.no_grad():
#                 for name, param in model.named_parameters():
#                     if param.grad is not None:
#                         grad_norm = param.grad.data.norm(2).item()
#                         grad_norms[name] = grad_norm
#                         grad_stats['total_norm'] += grad_norm ** 2
#
#                         if torch.isnan(param.grad).any():
#                             grad_stats['nan_count'] += 1
#                         if torch.isinf(param.grad).any():
#                             grad_stats['inf_count'] += 1
#
#                         if grad_norm > grad_stats['max_norm']:
#                             grad_stats['max_norm'] = grad_norm
#                             grad_stats['max_layer'] = name
#
#                         if grad_norm < grad_stats['min_norm']:
#                             grad_stats['min_norm'] = grad_norm
#                             grad_stats['min_layer'] = name
#
#                 grad_stats['total_norm'] = grad_stats['total_norm'] ** 0.5
#
#             optimizer.step()
#
#
#
#         torch.cuda.synchronize()
#         if model_ema is not None:
#             model_ema.update(model)
#
#         metric_logger.update(loss=loss_value)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#
#         # 添加以下代码↓↓↓
#         # ============== 梯度问题检测和日志记录 ==============
#         # 记录到metric_logger
#         metric_logger.update(grad_norm=grad_stats['total_norm'])
#         metric_logger.update(max_grad_norm=grad_stats['max_norm'])
#         metric_logger.update(min_grad_norm=grad_stats['min_norm'])
#
#         # 梯度问题检测
#         if grad_stats['total_norm'] > 1000:  # 梯度爆炸阈值
#             print(f"⚠️ Gradient explosion detected! Total norm: {grad_stats['total_norm']:.2f}")
#             # 记录最大梯度层
#             if args.local_rank == 0 and args.gpu == 0:
#                 mlflow.log_metric("grad_explosion/max_layer", grad_stats['max_layer'], step=metric_logger.step)
#
#         if grad_stats['total_norm'] < 1e-5:  # 梯度消失阈值
#             print(f"⚠️ Gradient vanishing detected! Total norm: {grad_stats['total_norm']:.2e}")
#
#         if grad_stats['nan_count'] > 0:
#             print(f"⚠️ NaN gradients detected in {grad_stats['nan_count']} layers!")
#
#         if grad_stats['inf_count'] > 0:
#             print(f"⚠️ Inf gradients detected in {grad_stats['inf_count']} layers!")
#
#         # MLflow日志记录
#         if args.local_rank == 0 and args.gpu == 0:
#             mlflow.log_metric("grad_norm/total", grad_stats['total_norm'], step=metric_logger.step)
#             mlflow.log_metric("grad_norm/max", grad_stats['max_norm'], step=metric_logger.step)
#             mlflow.log_metric("grad_norm/min", grad_stats['min_norm'], step=metric_logger.step)
#             mlflow.log_metric("grad_norm/nan_count", grad_stats['nan_count'], step=metric_logger.step)
#             mlflow.log_metric("grad_norm/inf_count", grad_stats['inf_count'], step=metric_logger.step)
#
#             # 每100步记录一次详细梯度分布
#             if metric_logger.step % 100 == 0:
#                 for name, norm in grad_norms.items():
#                     # 简化层名，避免MLflow中的命名过长问题
#                     simplified_name = name.replace('.', '_')[:50]
#                     mlflow.log_metric(f"grad_norms/{simplified_name}", norm, step=metric_logger.step)
#         # ===========================================
#
#
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
