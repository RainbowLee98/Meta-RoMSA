import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

from torch import optim
from torch import exp
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.models.openai.tokenization_openai import text_standardize
from models.subNets.BertTextEncoder import BertTextEncoder
from models.subNets.EMT import EMT
from transformers import BertModel

from models.subNets.trans import seqEncoder, meta_seqEncoder
from models.subNets.tools import init_net
from config.get_data_root import data_root

__all__ = ['META_ROMSA']


class Linear_fw(nn.Linear):
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast,
                           self.bias.fast)  # weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class PreUnit(nn.Module):
    def __init__(self, dropout, in_dim, mid_dim) -> None:
        super().__init__()
        self.post_dropout = nn.Dropout(dropout)
        self.post_layer_1 = Linear_fw(in_dim, mid_dim)
        self.post_layer_2 = Linear_fw(mid_dim, mid_dim)
        self.out = Linear_fw(mid_dim, 1)

    def forward(self, x):
        out = self.post_dropout(x)
        out = F.relu(self.post_layer_1(out), inplace=False)
        out = F.relu(self.post_layer_2(out), inplace=False)
        out = self.out(out)
        return out


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


# ds
class uniweight_Norm(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.sequential = nn.Sequential(
            Linear_fw(in_dim, mid_dim * 2),
            nn.LayerNorm(mid_dim * 2),  # 添加归一化层
            nn.ELU(inplace=True),  # 改用ELU激活
            nn.Dropout(0.3),
            Linear_fw(mid_dim * 2, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ELU(inplace=True),
            Linear_fw(mid_dim, out_dim)
        )
        # 初始化最后一层权重接近零
        nn.init.uniform_(self.sequential[-1].weight, -1e-5, 1e-5)
        nn.init.uniform_(self.sequential[-1].bias, -1e-5, 1e-5)

    def forward(self, x):
        # 数据集默认维度是 b,t,d 调整为b,d以适配后续
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        resize = nn.Linear(x_flat.shape[1], self.in_dim).to(x.device)
        x_resize = resize(x_flat)
        return torch.sigmoid(self.sequential(x_resize)) * 2  # 输出范围[0,2]增加灵活性


# class uniweight(nn.Module):
#     def __init__(self, in_dim, mid_dim, out_dim) -> None:
#         super().__init__()
#         self.in_dim = in_dim
#         self.linear1 = Linear_fw(in_dim, mid_dim)
#         self.relu = nn.ReLU(inplace=True)
#         self.linear2 = Linear_fw(mid_dim, out_dim)
#
#     def forward(self, x):
#         # 数据集默认维度是 b,t,d 调整为b,d以适配后续
#         batch_size = x.shape[0]
#         x_flat = x.view(batch_size, -1)
#         resize = nn.Linear(x_flat.shape[1], self.in_dim).to(x.device)
#         x_resize = resize(x_flat)
#
#         x = self.linear1(x_resize)
#         x = self.relu(x)
#         out = self.linear2(x)
#         return torch.sigmoid(out)

class uniweight(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.sequential = nn.Sequential(
            Linear_fw(in_dim, mid_dim),
            nn.ReLU(inplace=True),
            Linear_fw(mid_dim, out_dim)
        )
        # 初始化最后一层权重和偏置接近零
        nn.init.uniform_(self.sequential[-1].weight, -1e-5, 1e-5)
        nn.init.uniform_(self.sequential[-1].bias, -1e-5, 1e-5)

    def forward(self, x):
        # 调整输入维度：b,t,d -> b, (t*d)
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        # 将输入特征调整到指定维度
        resize = nn.Linear(x_flat.shape[1], self.in_dim).to(x.device)
        x_resized = resize(x_flat)
        # 通过Sequential处理并应用Sigmoid
        return torch.sigmoid(self.sequential(x_resized))


class META_ROMSA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        dataset_path = data_root

        # 编码器&调整维度
        self.net_trans_A = meta_seqEncoder(args.feature_dims[1], args.mid_dim, dropout=args.dropout_rate)
        self.net_trans_L = meta_seqEncoder(args.feature_dims[0], args.mid_dim, dropout=args.dropout_rate)
        self.net_trans_V = meta_seqEncoder(args.feature_dims[2], args.mid_dim, dropout=args.dropout_rate)

        self.criterion_l1 = nn.L1Loss()
        self.fast_parameters = []
        self.device = args.device
        self.aligned = args.need_data_aligned  # 是否需要对齐数据

        # ===== 模型组件分组 =====

        self.backbone_names = [
            'post_fusion', 'post_text', 'post_audio', 'post_video',
            't2a', 't2v', 'a2t', 'a2v', 'v2t', 'v2a',
            'net_trans_A', 'net_trans_L', 'net_trans_V',
        ]
        self.wnet_models = ['text_w_net', 'audio_w_net', 'video_w_net']

        self.model_names = self.backbone_names + self.wnet_models

        ########### 元学习 ###########
        # bert
        # 权重矩阵
        if args.datasetName == 'meta-sims' or args.datasetName == 'meta-simsv2':
            self.bert = BertModel.from_pretrained(os.path.join(dataset_path, 'metana_pretrained'),
                                                  local_files_only=True)
            self.text_w_net = uniweight(in_dim=args.output_dim * 3 + args.mid_dim * 3,
                                        mid_dim=128, out_dim=1)
            self.audio_w_net = uniweight(in_dim=args.output_dim * 3 + args.mid_dim * 3,
                                         mid_dim=128, out_dim=1)
            self.video_w_net = uniweight(in_dim=args.output_dim * 3 + args.mid_dim * 3,
                                         mid_dim=128, out_dim=1)
        else:
            self.bert = BertModel.from_pretrained(os.path.join(dataset_path, 'pretrained_berts/bert_en'),
                                                  local_files_only=True)
            self.text_w_net = uniweight_Norm(in_dim=args.output_dim * 3 + args.mid_dim * 3,
                                             mid_dim=128, out_dim=1)
            self.audio_w_net = uniweight_Norm(in_dim=args.output_dim * 3 + args.mid_dim * 3,
                                              mid_dim=128, out_dim=1)
            self.video_w_net = uniweight_Norm(in_dim=args.output_dim * 3 + args.mid_dim * 3,
                                              mid_dim=128, out_dim=1)
        self.bert = self.bert.to(self.device)
        self.bert.eval()

        self.criterion_attra = nn.CosineSimilarity(dim=1).cuda(self.args.gpu_ids)
        self.criterion_recon = ReconLoss(self.args.recon_loss)
        ########### 原始训练 ###########
        ## 文本编码器
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        ## 音频和视觉编码器
        audio_in, video_in = args.feature_dims[1:]  # 获取音频和视觉输入维度
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, num_layers=args.a_lstm_layers,
                                      dropout=args.a_lstm_dropout)
        self.video_model = AuViSubNet(video_in, args.v_lstm_hidden_size, args.video_out, num_layers=args.v_lstm_layers,
                                      dropout=args.v_lstm_dropout)
        # 特征投影层：将不同模态的特征投影到同一维度
        self.proj_audio = nn.Linear(args.audio_out, args.d_model,
                                    bias=False) if args.audio_out != args.d_model else nn.Identity()
        self.proj_video = nn.Linear(args.video_out, args.d_model,
                                    bias=False) if args.video_out != args.d_model else nn.Identity()
        self.proj_text = nn.Linear(args.text_out, args.d_model,
                                   bias=False) if args.text_out != args.d_model else nn.Identity()

        # 多模态融合：使用增强型多模态Transformer（EMT）
        num_modality = 3  # 模态数量（文本、音频、视觉）
        self.fusion = EMT(dim=args.d_model, depth=args.fusion_layers, heads=args.heads, num_modality=num_modality,
                          learnable_pos_emb=args.learnable_pos_emb, emb_dropout=args.emb_dropout,
                          attn_dropout=args.attn_dropout,
                          ff_dropout=args.ff_dropout, ff_expansion=args.ff_expansion, mpu_share=args.mpu_share,
                          modality_share=args.modality_share, layer_share=args.layer_share,
                          attn_act_fn=args.attn_act_fn)

        ########### 高层吸引与低层重建 ###########
        # 通过SimSiam进行高层特征吸引
        ## 投影器
        ## gmc_tokens: 全局多模态上下文
        gmc_tokens_dim = num_modality * args.d_model  # 全局多模态上下文的维度
        self.gmc_tokens_projector = Projector(gmc_tokens_dim, gmc_tokens_dim)  # 投影全局多模态上下文
        self.text_projector = Projector(args.text_out, args.text_out)  # 投影文本特征
        self.audio_projector = Projector(args.audio_out, args.audio_out)  # 投影音频特征
        self.video_projector = Projector(args.video_out, args.video_out)  # 投影视觉特征

        ## 预测器
        self.gmc_tokens_predictor = Predictor(gmc_tokens_dim, args.gmc_tokens_pred_dim, gmc_tokens_dim)  # 预测全局多模态上下文
        self.text_predictor = Predictor(args.text_out, args.text_pred_dim, args.text_out)  # 预测文本特征
        self.audio_predictor = Predictor(args.audio_out, args.audio_pred_dim, args.audio_out)  # 预测音频特征
        self.video_predictor = Predictor(args.video_out, args.video_pred_dim, args.video_out)  # 预测视觉特征

        # 低层特征重建
        self.recon_text = nn.Linear(args.d_model, args.feature_dims[0])  # 重建文本特征
        self.recon_audio = nn.Linear(args.d_model, args.feature_dims[1])  # 重建音频特征
        self.recon_video = nn.Linear(args.d_model, args.feature_dims[2])  # 重建视觉特征

        # 最终预测模块
        self.post_fusion = PreUnit(dropout=args.post_fusion_dropout,
                                   in_dim=args.text_out + args.video_out + args.audio_out + gmc_tokens_dim,
                                   mid_dim=args.post_fusion_dim)
        ########### 单模态预测 ###########
        self.post_text = PreUnit(dropout=args.post_fusion_dropout, in_dim=args.text_out, mid_dim=args.post_fusion_dim)
        self.post_audio = PreUnit(dropout=args.post_fusion_dropout, in_dim=args.audio_out, mid_dim=args.post_fusion_dim)
        self.post_video = PreUnit(dropout=args.post_fusion_dropout, in_dim=args.video_out, mid_dim=args.post_fusion_dim)

        ########### 模态翻译 ###########

        self.dropout_rate = args.dropout_rate

        self.a2t = seqEncoder(self.args.mid_dim, self.args.mid_dim, dropout=self.dropout_rate)
        self.v2t = seqEncoder(self.args.mid_dim, self.args.mid_dim, dropout=self.dropout_rate)

        self.t2a = seqEncoder(self.args.mid_dim, self.args.mid_dim, dropout=self.dropout_rate)
        self.t2v = seqEncoder(self.args.mid_dim, self.args.mid_dim, dropout=self.dropout_rate)

        self.a2v = seqEncoder(self.args.mid_dim, self.args.mid_dim, dropout=self.dropout_rate)
        self.v2a = seqEncoder(self.args.mid_dim, self.args.mid_dim, dropout=self.dropout_rate)

        self.set_alpha_model()
        self.optimizers = []
        # ===== 初始化优化器 =====
        if self.training:
            self.setup(args)
            # 骨干网络优化
            self.parameter = [{'params': getattr(self, net).parameters()} for net in self.backbone_names]
            self.backbone_optimizer = torch.optim.SGD(self.parameter, lr=args.lr, momentum=0.9,
                                                      weight_decay=args.weight_decay)

            # 权重网络优化
            self.uniweightnet_parameter = [{'params': getattr(self, net).parameters()} for net in self.wnet_models]
            self.uniweight_optimizer = torch.optim.Adam(self.uniweightnet_parameter, lr=args.uniweight_lr, betas=(args.beta1, 0.999),
                                                        weight_decay=args.weight_decay)

            # 骨干网络 的动态学习率
            parameters_alpha = [
                {'params': getattr(self, f"{net}_alpha").parameters(), 'lr': args.alpha_lr,
                 'betas': (args.beta1, 0.999),
                 'weight_decay': args.weight_decay} for net in self.backbone_names]

            self.optimizer_alpha = torch.optim.Adam(parameters_alpha)

    def setup(self, args):
        print(f'self.model_names {self.model_names}')
        for name in self.model_names:
            print(name)
            net = getattr(self, name)
            net = init_net(net, args.init_type, args.init_gain, args.gpu_ids)
            setattr(self, name, net)

    def set_alpha_model(self):
        for m in self.model_names:
            lrs = [torch.ones_like(p).to(self.device) * self.args.inner_lr for p in getattr(self, m).parameters()]
            lrs = nn.ParameterList([nn.Parameter(lr) for lr in lrs])
            setattr(self, f"{m}_alpha", lrs)

    def view_create(self, batch_data):
        vision = batch_data['vision'].to(self.args.device)
        audio = batch_data['audio'].to(self.args.device)
        text = batch_data['text'].to(self.args.device)
        # incomplete (missing) view
        vision_m = batch_data['vision_m'].to(self.args.device)
        audio_m = batch_data['audio_m'].to(self.args.device)
        text_m = batch_data['text_m'].to(self.args.device)
        # mask
        vision_missing_mask = batch_data['vision_missing_mask'].to(self.args.device)
        audio_missing_mask = batch_data['audio_missing_mask'].to(self.args.device)
        text_missing_mask = batch_data['text_missing_mask'].to(self.args.device)
        vision_mask = batch_data['vision_mask'].to(self.args.device)
        audio_mask = batch_data['audio_mask'].to(self.args.device)

        labels = batch_data['labels']['M'].to(self.args.device)

        if not self.args.need_data_aligned:
            audio_lengths = batch_data['audio_lengths'].to(self.args.device)
            vision_lengths = batch_data['vision_lengths'].to(self.args.device)
        else:
            audio_lengths, vision_lengths = 0, 0

        res = {
            'vision': vision,
            'audio': audio,
            'text': text,

            'vision_m': vision_m,
            'audio_m': audio_m,
            'text_m': text_m,

            'vision_missing_mask': vision_missing_mask,
            'audio_missing_mask': audio_missing_mask,
            'text_missing_mask': text_missing_mask,

            'vision_mask': vision_mask,
            'audio_mask': audio_mask,

            'labels': labels,

            'audio_lengths': audio_lengths,
            'vision_lengths': vision_lengths,

        }
        if batch_data['labels'].get('T') is not None:
            # print('单模态标签使用')
            labels_T = batch_data['labels']['T'].to(self.args.device)
            labels_A = batch_data['labels']['A'].to(self.args.device)
            labels_V = batch_data['labels']['V'].to(self.args.device)
            res.update({
                'labels_T': labels_T,
                'labels_A': labels_A,
                'labels_V': labels_V,
            })

        return res

    def inner_train(self, batch_data):
        view_res = self.view_create(batch_data)

        outputs = self.forward((view_res['text'], view_res['text_m']),
                               (view_res['audio'], view_res['audio_m'], view_res['audio_lengths']),
                               (view_res['vision'], view_res['vision_m'], view_res['vision_lengths'])
                               )

        # 计算损失
        ## 元学习损失

        cost_m = torch.abs(outputs['pred'].view(-1) - view_res['labels'].view(-1))
        cost_m = torch.reshape(cost_m, (len(cost_m), 1))

        cost_t = torch.abs(outputs['pred_text'].view(-1) - view_res['labels'].view(-1))
        cost_t = torch.reshape(cost_t, (len(cost_t), 1))

        cost_a = torch.abs(outputs['pred_audio'].view(-1) - view_res['labels'].view(-1))
        cost_a = torch.reshape(cost_a, (len(cost_a), 1))

        cost_v = torch.abs(outputs['pred_video'].view(-1) - view_res['labels'].view(-1))
        cost_v = torch.reshape(cost_v, (len(cost_v), 1))

        cost_w_t = torch.cat((view_res['labels'], outputs['pred'], outputs['pred_text'], outputs['fusion_t']), -1)
        cost_w_a = torch.cat((view_res['labels'], outputs['pred'], outputs['pred_audio'], outputs['fusion_a']), -1)
        cost_w_v = torch.cat((view_res['labels'], outputs['pred'], outputs['pred_video'], outputs['fusion_v']), -1)
        with torch.no_grad():
            text_w_lambda = self.text_w_net(cost_w_t.data)
            audio_w_lambda = self.audio_w_net(cost_w_a.data)
            video_w_lambda = self.video_w_net(cost_w_v.data)

        text_w_meta = torch.sum(cost_t * text_w_lambda) / len(cost_t)
        audio_w_meta = torch.sum(cost_a * audio_w_lambda) / len(cost_a)
        video_w_meta = torch.sum(cost_v * video_w_lambda) / len(cost_v)

        m_w_meta = torch.sum(cost_m) / len(cost_m)

        # loss_meta = m_w_meta + text_w_meta + audio_w_meta + video_w_meta
        loss_meta = m_w_meta + text_w_meta + (audio_w_meta + video_w_meta)*self.args.loss_meta_weight_unimodal



        ## 翻译损失
        # 翻译与循环一致性 损失
        loss_trans = self.criterion_l1(outputs['text_ori'], outputs['text_audio_text']) * self.args.trans_txt
        loss_trans += self.criterion_l1(outputs['text_ori'], outputs['text_vision_text']) * self.args.trans_txt
        loss_trans += self.criterion_l1(outputs['audio_ori'], outputs['audio_text_audio']) * self.args.trans_xtx
        loss_trans += self.criterion_l1(outputs['audio_ori'], outputs['audio_vision_audio']) * self.args.trans_xtx
        loss_trans += self.criterion_l1(outputs['vision_ori'], outputs['vision_text_vision']) * self.args.trans_xxx
        loss_trans += self.criterion_l1(outputs['vision_ori'], outputs['vision_audio_vision']) * self.args.trans_xxx

        loss_trans += self.criterion_l1(outputs['text_ori'], outputs['audio_text']) * self.args.trans_xt
        loss_trans += self.criterion_l1(outputs['text_ori'], outputs['vision_text']) * self.args.trans_xt
        loss_trans += self.criterion_l1(outputs['audio_ori'], outputs['text_audio']) * self.args.trans_tx
        loss_trans += self.criterion_l1(outputs['audio_ori'], outputs['vision_audio']) * self.args.trans_xt
        loss_trans += self.criterion_l1(outputs['vision_ori'], outputs['text_vision']) * self.args.trans_xx
        loss_trans += self.criterion_l1(outputs['vision_ori'], outputs['audio_vision']) * self.args.trans_xx

        ## 预测损失
        loss_pred_m = torch.mean(torch.abs(outputs['pred_m'].view(-1) - view_res['labels'].view(-1)))
        loss_pred = torch.mean(torch.abs(outputs['pred'].view(-1) - view_res['labels'].view(-1)))

        ## attraction loss (high-level) 高层吸引损失
        loss_attra_gmc_tokens = -(self.criterion_attra(outputs['p_gmc_tokens_m'], outputs['z_gmc_tokens']).mean() +
                                  self.criterion_attra(outputs['p_gmc_tokens'], outputs['z_gmc_tokens_m']).mean()) * 0.5
        loss_attra_text = -(self.criterion_attra(outputs['p_text_m'], outputs['z_text']).mean() +
                            self.criterion_attra(outputs['p_text'], outputs['z_text_m']).mean()) * 0.5
        loss_attra_audio = -(self.criterion_attra(outputs['p_audio_m'], outputs['z_audio']).mean() +
                             self.criterion_attra(outputs['p_audio'], outputs['z_audio_m']).mean()) * 0.5
        loss_attra_video = -(self.criterion_attra(outputs['p_video_m'], outputs['z_video']).mean() +
                             self.criterion_attra(outputs['p_video'], outputs['z_video_m']).mean()) * 0.5
        loss_attra = loss_attra_gmc_tokens + loss_attra_text + loss_attra_audio + loss_attra_video + loss_pred

        ## reconstruction loss (low-level) 低层重建损失
        mask = view_res['text'][:, 1, 1:] - view_res['text_missing_mask'][:, 1:]  # '1:' for excluding CLS
        loss_recon_text = self.criterion_recon(outputs['text_recon'], outputs['text_for_recon'], mask)

        mask = view_res['audio_mask'] - view_res['audio_missing_mask']
        loss_recon_audio = self.criterion_recon(outputs['audio_recon'],
                                                view_res['audio'][:, : view_res['audio_lengths'].max()],
                                                mask[:, : view_res['audio_lengths'].max()])

        mask = view_res['vision_mask'] - view_res['vision_missing_mask']
        loss_recon_video = self.criterion_recon(outputs['video_recon'],
                                                view_res['vision'][:, : view_res['vision_lengths'].max()],
                                                mask[:, : view_res['vision_lengths'].max()])
        loss_recon = loss_recon_text + loss_recon_audio + loss_recon_video

        ## total loss
        loss = loss_pred_m + self.args.loss_attra_weight * loss_attra + self.args.loss_recon_weight * loss_recon + loss_trans + loss_meta * self.args.loss_meta_weight

        self.zero_grad()
        self.backward(loss)

        self.optimizer_alpha.step()

        pred_m = outputs['pred_m']
        labels = view_res['labels']
        res = {
            'loss': loss,
            'loss_pred_m': loss_pred_m,
            'loss_attra': loss_attra,
            'loss_recon': loss_recon,
            'loss_trans': loss_trans,
            'loss_meta': loss_meta,
            'loss_meta_multi': m_w_meta,
            'loss_meta_text': text_w_meta,
            'loss_meta_audio': audio_w_meta,
            'loss_meta_vision': video_w_meta,
            'labels': labels,
            'pred_m': pred_m,
        }
        del outputs
        del view_res

        return res

    def support_train(self, batch_data):

        view_res = self.view_create(batch_data)

        outputs = self.forward((view_res['text'], view_res['text_m']),
                               (view_res['audio'], view_res['audio_m'], view_res['audio_lengths']),
                               (view_res['vision'], view_res['vision_m'], view_res['vision_lengths'])
                               )

        # 计算损失
        ## 元学习损失

        cost_m = torch.abs(outputs['pred'].view(-1) - view_res['labels'].view(-1))
        cost_m = torch.reshape(cost_m, (len(cost_m), 1))

        cost_t = torch.abs(outputs['pred_text'].view(-1) - view_res['labels'].view(-1))
        cost_t = torch.reshape(cost_t, (len(cost_t), 1))

        cost_a = torch.abs(outputs['pred_audio'].view(-1) - view_res['labels'].view(-1))
        cost_a = torch.reshape(cost_a, (len(cost_a), 1))

        cost_v = torch.abs(outputs['pred_video'].view(-1) - view_res['labels'].view(-1))
        cost_v = torch.reshape(cost_v, (len(cost_v), 1))

        cost_w_t = torch.cat((view_res['labels'], outputs['pred'], outputs['pred_text'], outputs['fusion_t']), -1)
        cost_w_a = torch.cat((view_res['labels'], outputs['pred'], outputs['pred_audio'], outputs['fusion_a']), -1)
        cost_w_v = torch.cat((view_res['labels'], outputs['pred'], outputs['pred_video'], outputs['fusion_v']), -1)

        text_w_lambda = self.text_w_net(cost_w_t.data)
        text_w_meta = torch.sum(cost_t * text_w_lambda) / len(cost_t)

        audio_w_lambda = self.audio_w_net(cost_w_a.data)
        audio_w_meta = torch.sum(cost_a * audio_w_lambda) / len(cost_a)

        video_w_lambda = self.video_w_net(cost_w_v.data)
        video_w_meta = torch.sum(cost_v * video_w_lambda) / len(cost_v)

        m_w_meta = torch.sum(cost_m) / len(cost_m)

        # loss_meta = m_w_meta + text_w_meta + (audio_w_meta + video_w_meta)*self.args.loss_meta_weight_unimodal
        loss_meta = m_w_meta + text_w_meta + audio_w_meta + video_w_meta


        ## 翻译损失
        # 翻译与循环一致性 损失
        loss_trans = self.criterion_l1(outputs['text_ori'], outputs['text_audio_text']) * self.args.trans_txt
        loss_trans += self.criterion_l1(outputs['text_ori'], outputs['text_vision_text']) * self.args.trans_txt
        loss_trans += self.criterion_l1(outputs['audio_ori'], outputs['audio_text_audio']) * self.args.trans_xtx
        loss_trans += self.criterion_l1(outputs['audio_ori'], outputs['audio_vision_audio']) * self.args.trans_xtx
        loss_trans += self.criterion_l1(outputs['vision_ori'], outputs['vision_text_vision']) * self.args.trans_xxx
        loss_trans += self.criterion_l1(outputs['vision_ori'], outputs['vision_audio_vision']) * self.args.trans_xxx

        loss_trans += self.criterion_l1(outputs['text_ori'], outputs['audio_text']) * self.args.trans_xt
        loss_trans += self.criterion_l1(outputs['text_ori'], outputs['vision_text']) * self.args.trans_xt
        loss_trans += self.criterion_l1(outputs['audio_ori'], outputs['text_audio']) * self.args.trans_tx
        loss_trans += self.criterion_l1(outputs['audio_ori'], outputs['vision_audio']) * self.args.trans_xt
        loss_trans += self.criterion_l1(outputs['vision_ori'], outputs['text_vision']) * self.args.trans_xx
        loss_trans += self.criterion_l1(outputs['vision_ori'], outputs['audio_vision']) * self.args.trans_xx

        ## 预测损失
        loss_pred_m = torch.mean(torch.abs(outputs['pred_m'].view(-1) - view_res['labels'].view(-1)))
        loss_pred = torch.mean(torch.abs(outputs['pred'].view(-1) - view_res['labels'].view(-1)))

        ## attraction loss (high-level) 高层吸引损失
        loss_attra_gmc_tokens = -(self.criterion_attra(outputs['p_gmc_tokens_m'], outputs['z_gmc_tokens']).mean() +
                                  self.criterion_attra(outputs['p_gmc_tokens'], outputs['z_gmc_tokens_m']).mean()) * 0.5
        loss_attra_text = -(self.criterion_attra(outputs['p_text_m'], outputs['z_text']).mean() +
                            self.criterion_attra(outputs['p_text'], outputs['z_text_m']).mean()) * 0.5
        loss_attra_audio = -(self.criterion_attra(outputs['p_audio_m'], outputs['z_audio']).mean() +
                             self.criterion_attra(outputs['p_audio'], outputs['z_audio_m']).mean()) * 0.5
        loss_attra_video = -(self.criterion_attra(outputs['p_video_m'], outputs['z_video']).mean() +
                             self.criterion_attra(outputs['p_video'], outputs['z_video_m']).mean()) * 0.5
        loss_attra = loss_attra_gmc_tokens + loss_attra_text + loss_attra_audio + loss_attra_video + loss_pred

        ## reconstruction loss (low-level) 低层重建损失
        mask = view_res['text'][:, 1, 1:] - view_res['text_missing_mask'][:, 1:]  # '1:' for excluding CLS
        loss_recon_text = self.criterion_recon(outputs['text_recon'], outputs['text_for_recon'], mask)

        mask = view_res['audio_mask'] - view_res['audio_missing_mask']
        loss_recon_audio = self.criterion_recon(outputs['audio_recon'],
                                                view_res['audio'][:, : view_res['audio_lengths'].max()],
                                                mask[:, : view_res['audio_lengths'].max()])

        mask = view_res['vision_mask'] - view_res['vision_missing_mask']
        loss_recon_video = self.criterion_recon(outputs['video_recon'],
                                                view_res['vision'][:, : view_res['vision_lengths'].max()],
                                                mask[:, : view_res['vision_lengths'].max()])
        loss_recon = loss_recon_text + loss_recon_audio + loss_recon_video

        ## total loss
        loss = loss_pred_m + self.args.loss_attra_weight * loss_attra + self.args.loss_recon_weight * loss_recon + loss_trans + loss_meta * self.args.support_loss_meta_weight

        grad = torch.autograd.grad(loss, self.fast_parameters, create_graph=True)

        self.fast_parameters = []

        self.update_pra(grad)

        del grad
        del outputs
        del view_res

    def query_train(self, model, query_batch):

        view_res = self.view_create(query_batch)

        outputs = self.query_forward((view_res['text'], view_res['text_m']),
                                     (view_res['audio'], view_res['audio_m'], view_res['audio_lengths']),
                                     (view_res['vision'], view_res['vision_m'], view_res['vision_lengths']))

        m_meta = F.smooth_l1_loss(outputs['pred'], view_res['labels'], reduction='none')
        t_meta = F.smooth_l1_loss(outputs['pred_text'], view_res['labels_T'], reduction='none')
        a_meta = F.smooth_l1_loss(outputs['pred_audio'], view_res['labels_A'], reduction='none')
        v_meta = F.smooth_l1_loss(outputs['pred_video'], view_res['labels_V'], reduction='none')
        # 这里和 支持集训练 有区别，不需要加权重
        m_meta = torch.mean(m_meta)
        t_meta = torch.mean(t_meta)
        a_meta = torch.mean(a_meta)
        v_meta = torch.mean(v_meta)

        query_losses = m_meta + t_meta + (a_meta + v_meta) * self.args.loss_meta_weight_unimodal

        self.wnet_zero_grad()
        query_losses.backward()
        self.uniweight_optimizer.step()

        del outputs
        del view_res

    def update_pra(self, grad):
        pointer = 0
        for m in self.backbone_names:
            for weight, alpha in zip(getattr(self, m).parameters(), getattr(self, f"{m}_alpha").parameters()):
                # alpha is limited to 0 to 0.03
                alpha = special_sigmoid(alpha)

                if weight.fast is None:
                    weight.fast = weight - torch.mul(alpha, grad[pointer])  # create weight.fast
                else:
                    weight.fast = weight.fast - torch.mul(alpha, grad[pointer])  # update weight.fast
                pointer += 1
                self.fast_parameters.append(
                    weight.fast)  # gradients are based on newest weights, but the graph will retain the link to old weight.fasts

    def net_reset(self):
        self.fast_parameters = self.get_inner_loop_params()
        for m in self.backbone_names:
            for weight in getattr(self, m).parameters():  # reset fast parameters
                weight.fast = None

    def get_inner_loop_params(self):
        params = []
        for name in self.backbone_names:
            if hasattr(self, name):
                params += list(getattr(self, name).parameters())
        return params

    def backward(self, losses):
        losses.backward()
        for m in self.backbone_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, m).parameters(), 2)

    def zero_grad(self):
        for m in self.backbone_names:
            getattr(self, m).zero_grad()
            getattr(self, f"{m}_alpha").zero_grad()

    def wnet_zero_grad(self):
        for m in self.wnet_models:
            getattr(self, m).zero_grad()

    def forward_once(self, text, text_lengths, audio, audio_lengths, video, video_lengths, missing):
        # 翻译
        text_bert = text
        with torch.no_grad():
            text1 = self.bert(
                input_ids=text_bert[:, 0, :].long(),
                attention_mask=text_bert[:, 1, :].long(),
                token_type_ids=text_bert[:, 2, :].long())[0]

        text_ori = self.net_trans_L(text1)
        audio_ori = self.net_trans_A(audio)
        vision_ori = self.net_trans_V(video)

        text_audio = self.t2a(text_ori)
        text_vision = self.t2v(text_ori)
        audio_text = self.a2t(audio_ori)
        audio_vision = self.a2v(audio_ori)
        vision_text = self.v2t(vision_ori)
        vision_audio = self.v2a(vision_ori)

        text_audio_text = self.a2t(text_audio)
        text_vision_text = self.v2t(text_vision)
        audio_text_audio = self.t2a(audio_text)
        audio_vision_audio = self.v2a(audio_vision)
        vision_text_vision = self.t2v(vision_text)
        vision_audio_vision = self.a2v(vision_audio)

        fusion_t = torch.cat((text_ori, text_audio, text_vision), 1)
        fusion_a = torch.cat((audio_ori, audio_text, audio_vision), 1)
        fusion_v = torch.cat((vision_ori, vision_text, vision_audio), 1)

        # unimodal encoders
        text = self.text_model(text)
        text_utt, text = text[:, 0], text[:, 1:]  # (B, 1, D), (B, T, D)
        text_for_recon = text.detach()

        audio, audio_utt = self.audio_model(audio, audio_lengths, return_temporal=True)
        video, video_utt = self.video_model(video, video_lengths, return_temporal=True)

        # projection
        ## gmc_tokens: global multimodal context, (B, 3, D)
        gmc_tokens = torch.stack([self.proj_text(text_utt), self.proj_audio(audio_utt), self.proj_video(video_utt)],
                                 dim=1)
        ## local unimodal features, (B, T, D)
        text, audio, video = self.proj_text(text), self.proj_audio(audio), self.proj_video(video)

        # get attention mask
        modality_masks = [length_to_mask(seq_len, max_len=max_len)
                          for seq_len, max_len in zip([text_lengths, audio_lengths, video_lengths],
                                                      [text.shape[1], audio.shape[1], video.shape[1]])]

        # fusion
        gmc_tokens, modality_ouputs = self.fusion(gmc_tokens, [text, audio, video], modality_masks)
        gmc_tokens = gmc_tokens.reshape(gmc_tokens.shape[0], -1)  # (B, 3*D)

        # high-level feature attraction via SimSiam
        ## projector
        z_gmc_tokens = self.gmc_tokens_projector(gmc_tokens)
        z_text = self.text_projector(text_utt)
        z_audio = self.audio_projector(audio_utt)
        z_video = self.video_projector(video_utt)

        ## predictor
        p_gmc_tokens = self.gmc_tokens_predictor(z_gmc_tokens)
        p_text = self.text_predictor(z_text)
        p_audio = self.audio_predictor(z_audio)
        p_video = self.video_predictor(z_video)

        # final prediction module
        fusion_h = torch.cat([text_utt, audio_utt, video_utt, gmc_tokens], dim=-1)
        output_fusion = self.post_fusion(fusion_h)

        #### 单模态预测
        ## 文本预测
        pred_text = self.post_text(text_utt)
        pred_audio = self.post_audio(audio_utt)
        pred_video = self.post_video(video_utt)

        suffix = '_m' if missing else ''
        res = {
            f'pred{suffix}': output_fusion,
            f'z_gmc_tokens{suffix}': z_gmc_tokens.detach(),
            f'p_gmc_tokens{suffix}': p_gmc_tokens,
            f'z_text{suffix}': z_text.detach(),
            f'p_text{suffix}': p_text,
            f'z_audio{suffix}': z_audio.detach(),
            f'p_audio{suffix}': p_audio,
            f'z_video{suffix}': z_video.detach(),
            f'p_video{suffix}': p_video,
        }
        # 单模态预测结果
        res.update({
            f'pred_text{suffix}': pred_text,
            f'pred_audio{suffix}': pred_audio,
            f'pred_video{suffix}': pred_video,
        })

        # 翻译结果
        res.update({
            f'fusion_t': fusion_t,
            f'fusion_a': fusion_a,
            f'fusion_v': fusion_v,

            f'text_ori': text_ori,
            f'audio_ori': audio_ori,
            f'vision_ori': vision_ori,

            f'text_audio': text_audio,
            f'text_vision': text_vision,
            f'audio_text': audio_text,
            f'audio_vision': audio_vision,
            f'vision_text': vision_text,
            f'vision_audio': vision_audio,
            f'text_audio_text': text_audio_text,
            f'text_vision_text': text_vision_text,
            f'audio_text_audio': audio_text_audio,
            f'audio_vision_audio': audio_vision_audio,
            f'vision_text_vision': vision_text_vision,
            f'vision_audio_vision': vision_audio_vision,

        })

        # low-level feature reconstruction
        if missing:
            text_recon = self.recon_text(modality_ouputs[0])
            audio_recon = self.recon_audio(modality_ouputs[1])
            video_recon = self.recon_video(modality_ouputs[2])
            res.update(
                {
                    'text_recon': text_recon,
                    'audio_recon': audio_recon,
                    'video_recon': video_recon,
                }
            )
        else:
            res.update({'text_for_recon': text_for_recon})

        return res

    def forward(self, text, audio, video):
        # 分离完整视图和缺失视图的输入
        text, text_m = text
        audio, audio_m, audio_lengths = audio
        video, video_m, video_lengths = video

        # 计算文本长度
        mask_len = torch.sum(text[:, 1, :], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach() - 2  # 减去CLS和SEP标记的长度

        # 完整视图的前向传播
        res = self.forward_once(text, text_lengths, audio, audio_lengths, video, video_lengths, missing=False)
        # 缺失视图的前向传播
        res_m = self.forward_once(text_m, text_lengths, audio_m, audio_lengths, video_m, video_lengths, missing=True)

        return {**res, **res_m}  # 返回完整视图和缺失视图的结果

    def query_forward_once(self, text, text_lengths, audio, audio_lengths, video, video_lengths, missing):

        # unimodal encoders
        text = self.text_model(text)
        text_utt, text = text[:, 0], text[:, 1:]  # (B, 1, D), (B, T, D)
        audio, audio_utt = self.audio_model(audio, audio_lengths, return_temporal=True)
        video, video_utt = self.video_model(video, video_lengths, return_temporal=True)

        # projection
        ## gmc_tokens: global multimodal context, (B, 3, D)
        gmc_tokens = torch.stack([self.proj_text(text_utt), self.proj_audio(audio_utt), self.proj_video(video_utt)],
                                 dim=1)
        ## local unimodal features, (B, T, D)
        text, audio, video = self.proj_text(text), self.proj_audio(audio), self.proj_video(video)

        # get attention mask
        modality_masks = [length_to_mask(seq_len, max_len=max_len)
                          for seq_len, max_len in zip([text_lengths, audio_lengths, video_lengths],
                                                      [text.shape[1], audio.shape[1], video.shape[1]])]

        # fusion
        gmc_tokens, modality_ouputs = self.fusion(gmc_tokens, [text, audio, video], modality_masks)
        gmc_tokens = gmc_tokens.reshape(gmc_tokens.shape[0], -1)  # (B, 3*D)
        # final prediction module
        fusion_h = torch.cat([text_utt, audio_utt, video_utt, gmc_tokens], dim=-1)
        output_fusion = self.post_fusion(fusion_h)

        #### 单模态预测
        pred_text = self.post_text(text_utt)
        pred_audio = self.post_audio(audio_utt)
        pred_video = self.post_video(video_utt)

        suffix = '_m' if missing else ''

        # 单模态预测结果
        res = {
            f'pred{suffix}': output_fusion,
            f'pred_text{suffix}': pred_text,
            f'pred_audio{suffix}': pred_audio,
            f'pred_video{suffix}': pred_video,
        }
        return res

    def query_forward(self, text, audio, video):
        # 分离完整视图和缺失视图的输入
        text, text_m = text
        audio, audio_m, audio_lengths = audio
        video, video_m, video_lengths = video

        # 计算文本长度
        mask_len = torch.sum(text[:, 1, :], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach() - 2  # 减去CLS和SEP标记的长度

        # 完整视图的前向传播
        res = self.query_forward_once(text, text_lengths, audio, audio_lengths, video, video_lengths, missing=False)
        # 缺失视图的前向传播
        res_m = self.query_forward_once(text_m, text_lengths, audio_m, audio_lengths, video_m, video_lengths,
                                        missing=True)

        return {**res, **res_m}  # 返回完整视图和缺失视图的结果


class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size=None, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: 输入维度
            hidden_size: 隐藏层维度
            num_layers: LSTM的层数
            dropout: dropout概率
            bidirectional: 是否使用双向LSTM
        Output:
            (前向传播返回值) 形状为(batch_size, out_size)的张量
        '''
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                           batch_first=True)  # LSTM层
        self.dropout = nn.Dropout(dropout)  # Dropout层
        feature_size = hidden_size * 2 if bidirectional else hidden_size  # 特征维度
        self.linear_1 = nn.Linear(feature_size,
                                  out_size) if feature_size != out_size and out_size is not None else nn.Identity()  # 全连接层

    def forward(self, x, lengths, return_temporal=False):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        # 将变长序列打包
        packed_sequence = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_last_hidden_state, final_states = self.rnn(packed_sequence)  # LSTM前向传播

        h = self.dropout(final_states[0].squeeze())  # 应用Dropout
        y_1 = self.linear_1(h)  # 全连接层
        if not return_temporal:
            return y_1  # 返回句子级特征
        else:
            unpacked_last_hidden_state, _ = pad_packed_sequence(packed_last_hidden_state, batch_first=True)  # 解包序列
            last_hidden_state = self.linear_1(unpacked_last_hidden_state)  # 获取时间级特征
            return last_hidden_state, y_1  # 返回时间级特征和句子级特征


class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, output_dim),
                                 nn.BatchNorm1d(output_dim, affine=False))  # 投影网络

    def forward(self, x):
        return self.net(x)  # 投影前向传播


class Predictor(nn.Module):
    def __init__(self, input_dim, pred_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, pred_dim, bias=False),
                                 nn.BatchNorm1d(pred_dim),
                                 nn.ReLU(inplace=True),  # 隐藏层
                                 nn.Linear(pred_dim, output_dim))  # 输出层

    def forward(self, x):
        return self.net(x)  # 预测前向传播


def special_sigmoid(x):
    """
        Calculate the value of the Sigmoid function
    """
    return 0.003 / (1 + exp(-500 * x))


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()  # 计算最大长度
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)  # 生成掩码
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)  # 转换数据类型
    return mask
