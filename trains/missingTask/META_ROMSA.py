import logging

from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop

logger = logging.getLogger('MSA')


class ReconLoss(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.eps = 1e-6
        self.type = type
        if type == 'L1Loss':
            self.loss = nn.L1Loss(reduction='sum')
        elif type == 'SmoothL1Loss':
            self.loss = nn.SmoothL1Loss(reduction='sum')
        elif type == 'MSELoss':
            self.loss = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError

    def forward(self, pred, target, mask):
        """
            pred, target -> batch, seq_len, d
            mask -> batch, seq_len
        """
        mask = mask.unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2]).float()

        loss = self.loss(pred * mask, target * mask) / (torch.sum(mask) + self.eps)

        return loss


class CountParam():
    # 统计模型参数
    def count_parameters_emt(self, model, name='.fusion'):
        answer = 0
        for n, p in model.named_parameters():
            if name in n:
                answer += p.numel()
        return answer

    # 统计融合模块 emt 参数
    def count_parameters(self, model):
        answer = 0
        for n, p in model.named_parameters():
            if 'predictor' not in n and 'projector' not in n and 'recon' not in n:
                answer += p.numel()
        return answer

    def print_param(self, model):
        logger.info(f'The model during inference has {self.count_parameters(model)} parameters.')
        logger.info(f'The fusion module (emt) has {self.count_parameters_emt(model)} parameters.')


class META_ROMSA():
    def __init__(self, args):
        assert args.train_mode == 'regression'
        # 参数
        self.args = args
        self.device = args.device
        self.args.tasks = "M"
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)
        self.criterion_l1 = nn.L1Loss()

        # criterion 高层吸引 低层重建损失
        self.criterion_attra = nn.CosineSimilarity(dim=1).cuda(self.args.gpu_ids)
        self.criterion_recon = ReconLoss(self.args.recon_loss)

        self.left_epochs = self.args.update_epochs

    def set_input(self, input):
        self.text = input['text'].float().to(self.args.device)
        self.audio = input['audio'].float().to(self.args.device)
        self.vision = input['vision'].float().to(self.args.device)
        if 't_label' in input:
            self.t_label = input['t_label'].type(torch.int64).to(self.args.device)
            self.a_label = input['a_label'].type(torch.int64).to(self.args.device)
            self.v_label = input['v_label'].type(torch.int64).to(self.args.device)

    def do_train(self, model, dataloader):

        # 计算参数量
        countparam = CountParam()
        countparam.print_param(model)

        # 参数优化
        # 衰减参数
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and \
                              'audio_model' not in n and 'video_model' not in n]

        # adam 优化参数
        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert,
             'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0,
             'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio,
             'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video,
             'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other,
             'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)

        # 初始化计数与结果 results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss', 'Loss(pred_m)'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        while True:
            epochs += 1
            # train
            y_pred, y_true = [], []
            # y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            # y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            s_t = time.time()

            # 元学习
            meta_loader_iter = iter(dataloader['query'])

            with tqdm(dataloader['train']) as td:
                for batch_idx, batch_data in enumerate(td, 1):
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    # 网络重置
                    model.Model.net_reset()
                    # 支持集训练
                    model.Model.support_train(batch_data)

                    # 验证集数据循环
                    try:
                        meta_batch = next(meta_loader_iter)
                    except StopIteration:
                        meta_loader_iter = iter(dataloader['query'])
                        meta_batch = next(meta_loader_iter)
                    # query集
                    self.set_input(meta_batch)
                    model.Model.query_train(model, meta_batch)

                    # 网络重置
                    model.Model.net_reset()
                    # 训练集
                    self.set_input(batch_data)
                    outputs = model.Model.inner_train(batch_data)

                    y_pred.append(outputs['pred_m'
                                          ''].cpu())
                    y_true.append(outputs['labels'].cpu())
                    loss = outputs['loss']

                    # 训练到一半输出一下
                    if batch_idx % (len(td) // 2) == 0:
                        logger.info(f'Epoch {epochs} | Batch {batch_idx:>3d} | [Train] Loss {loss:.4f}')
                    train_loss += loss.item()

                    # update parameters ，实现的梯度累计
                    if not left_epochs:
                        # update
                        optimizer.step()
                        left_epochs = self.args.update_epochs

                if not left_epochs:
                    # update
                    optimizer.step()

            e_t = time.time()
            logger.info(f'One epoch time for training: {e_t - s_t:.3f}s.')

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)

            # 创建空列表，为日志块提供固定结构。
            log_infos = [''] * 8
            # 分割线
            log_infos[0] = log_infos[-1] = '-' * 100

            # validation
            s_t = time.time()

            val_results = self.do_test(model, dataloader['valid'], mode="VAL", epochs=epochs)

            e_t = time.time()
            logger.info(f'One epoch time for validation: {e_t - s_t:.3f}s.')

            cur_valid = val_results[self.args.KeyEval]

            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                self.best_epoch = best_epoch
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
                log_infos[5] = f'==> Note: achieve best [Val] results at epoch {best_epoch}'

            log_infos[1] = f"Seed {self.args.seed} ({self.args.seeds.index(self.args.seed) + 1}/{self.args.num_seeds}) " \
                           f"| Epoch {epochs} (early stop={epochs - best_epoch}) | Train Loss {train_loss:.4f} | Val Loss {val_results['Loss']:.4f}"
            log_infos[2] = f"[Train] {dict_to_str(train_results)}"
            log_infos[3] = f"  [Val] {dict_to_str(val_results)}"

            # log information
            for log_info in log_infos:
                if log_info: logger.info(log_info)

            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                logger.info(
                    f"==> Note: since '{self.args.KeyEval}' does not improve in the past {self.args.early_stop} epochs, early stop the training process!")
                return

    def do_test(self, model, dataloader, criterion_attra=None, criterion_recon=None, mode="VAL", epochs=None):

        if epochs is None: logger.info("=" * 30 + f"Start Test of Seed {self.args.seed}" + "=" * 30)
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        eval_loss_pred = 0.0
        # criterion = nn.L1Loss()
        if criterion_attra is None: criterion_attra = nn.CosineSimilarity(dim=1).cuda(self.args.gpu_ids)
        if criterion_recon is None: criterion_recon = ReconLoss(self.args.recon_loss)
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_idx, batch_data in enumerate(td, 1):
                    # complete view
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    # incomplete (missing) view
                    vision_m = batch_data['vision_m'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device)
                    text_m = batch_data['text_m'].to(self.args.device)
                    vision_missing_mask = batch_data['vision_missing_mask'].to(self.args.device)
                    audio_missing_mask = batch_data['audio_missing_mask'].to(self.args.device)
                    text_missing_mask = batch_data['text_missing_mask'].to(self.args.device)
                    vision_mask = batch_data['vision_mask'].to(self.args.device)
                    audio_mask = batch_data['audio_mask'].to(self.args.device)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    labels = batch_data['labels']['M'].to(self.args.device).view(-1)
                    outputs = model((text, text_m), (audio, audio_m, audio_lengths), (vision, vision_m, vision_lengths))

                    # compute loss
                    # trans loss
                    # 循环一致性
                    loss_trans = self.criterion_l1(outputs['text_ori'],
                                                   outputs['text_audio_text']) * self.args.trans_txt
                    loss_trans += self.criterion_l1(outputs['text_ori'],
                                                    outputs['text_vision_text']) * self.args.trans_txt
                    loss_trans += self.criterion_l1(outputs['audio_ori'],
                                                    outputs['audio_text_audio']) * self.args.trans_xtx
                    loss_trans += self.criterion_l1(outputs['audio_ori'],
                                                    outputs['audio_vision_audio']) * self.args.trans_xtx
                    loss_trans += self.criterion_l1(outputs['vision_ori'],
                                                    outputs['vision_text_vision']) * self.args.trans_xxx
                    loss_trans += self.criterion_l1(outputs['vision_ori'],
                                                    outputs['vision_audio_vision']) * self.args.trans_xxx

                    loss_trans += self.criterion_l1(outputs['text_ori'], outputs['audio_text']) * self.args.trans_xt
                    loss_trans += self.criterion_l1(outputs['text_ori'], outputs['vision_text']) * self.args.trans_xt
                    loss_trans += self.criterion_l1(outputs['audio_ori'], outputs['text_audio']) * self.args.trans_tx
                    loss_trans += self.criterion_l1(outputs['audio_ori'], outputs['vision_audio']) * self.args.trans_xt
                    loss_trans += self.criterion_l1(outputs['vision_ori'], outputs['text_vision']) * self.args.trans_xx
                    loss_trans += self.criterion_l1(outputs['vision_ori'], outputs['audio_vision']) * self.args.trans_xx

                    ## task loss (prediction loss of incomplete view)
                    loss_pred_m = torch.mean(torch.abs(outputs['pred_m'].view(-1) - labels.view(-1)))
                    ## attraction loss (high-level)
                    loss_attra_gmc_tokens = -(
                            criterion_attra(outputs['p_gmc_tokens_m'], outputs['z_gmc_tokens']).mean() +
                            criterion_attra(outputs['p_gmc_tokens'], outputs['z_gmc_tokens_m']).mean()) * 0.5
                    loss_attra_text = -(criterion_attra(outputs['p_text_m'], outputs['z_text']).mean() +
                                        criterion_attra(outputs['p_text'], outputs['z_text_m']).mean()) * 0.5
                    loss_attra_audio = -(criterion_attra(outputs['p_audio_m'], outputs['z_audio']).mean() +
                                         criterion_attra(outputs['p_audio'], outputs['z_audio_m']).mean()) * 0.5
                    loss_attra_video = -(criterion_attra(outputs['p_video_m'], outputs['z_video']).mean() +
                                         criterion_attra(outputs['p_video'], outputs['z_video_m']).mean()) * 0.5
                    loss_pred = torch.mean(
                        torch.abs(outputs['pred'].view(-1) - labels.view(-1)))  # prediction loss of complete view
                    loss_attra = loss_attra_gmc_tokens + loss_attra_text + loss_attra_audio + loss_attra_video + loss_pred
                    ## reconstruction loss (low-level)
                    mask = text[:, 1, 1:] - text_missing_mask[:, 1:]  # '1:' for excluding CLS
                    loss_recon_text = criterion_recon(outputs['text_recon'], outputs['text_for_recon'], mask)
                    mask = audio_mask - audio_missing_mask
                    loss_recon_audio = criterion_recon(outputs['audio_recon'],
                                                       audio[:, : batch_data['audio_lengths'].max()],
                                                       mask[:, : batch_data['audio_lengths'].max()])
                    mask = vision_mask - vision_missing_mask
                    loss_recon_video = criterion_recon(outputs['video_recon'],
                                                       vision[:, : batch_data['vision_lengths'].max()],
                                                       mask[:, : batch_data['vision_lengths'].max()])
                    loss_recon = loss_recon_text + loss_recon_audio + loss_recon_video
                    ## total loss
                    loss = loss_pred_m + self.args.loss_attra_weight * loss_attra + self.args.loss_recon_weight * loss_recon + loss_trans * 0.5

                    if batch_idx % (len(td) // 2) == 0:
                        logger.info(f'Epoch {epochs} | Batch {batch_idx:>3d} | [Val] Loss {loss:.4f}')
                    eval_loss += loss.item()
                    eval_loss_pred += loss_pred_m.item()
                    y_pred.append(outputs['pred_m'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        eval_loss_pred = eval_loss_pred / len(dataloader)
        # logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        # logger.info('M: >> ' + dict_to_str(eval_results))
        eval_results['Loss'] = eval_loss
        eval_results['Loss(pred_m)'] = eval_loss_pred
        if epochs is None:  # for TEST
            logger.info(f"\n [Test] {dict_to_str(eval_results)}")
            logger.info(
                f"==> Note: achieve this results at epoch {self.best_epoch} (best [Val]) / {getattr(self, 'best_test_epoch', None)} (best [Test])")

        return eval_results
