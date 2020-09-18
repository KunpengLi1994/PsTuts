import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict


import torch.nn.functional as F
from GCN_lib.Res_GCN import Res_GCN

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, finetune=False,
                 cnn_type='vgg19', use_abs=False, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """

    img_enc = EncoderImagePrecomp(
        img_dim, embed_size, use_abs, no_imgnorm)


    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc_visual = nn.Linear(2048, embed_size)
        self.fc_popup = nn.Linear(2048, embed_size)
        self.fc_tool = nn.Linear(39, embed_size)

        self.init_weights()

        self.GCN_visual = Res_GCN(in_channels=embed_size, inter_channels=embed_size, sub_sample=True)
        self.GCN_popup = Res_GCN(in_channels=embed_size, inter_channels=embed_size, sub_sample=True)

        self.GCN_feat = Res_GCN(in_channels=embed_size, inter_channels=embed_size, sub_sample=True)

        self.feat_attn = nn.Linear(embed_size*3, 3)


    def weighted_feature(self, feat, attn):
        feat = feat.permute(0, 2, 1)
        attn = torch.unsqueeze(attn, 2)
        attn_feat = torch.matmul(feat, attn)
        attn_feat = torch.squeeze(attn_feat)

        return attn_feat


    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc_visual.in_features +
                                  self.fc_visual.out_features)
        self.fc_visual.weight.data.uniform_(-r, r)
        self.fc_visual.bias.data.fill_(0)


        r = np.sqrt(6.) / np.sqrt(self.fc_popup.in_features +
                                  self.fc_popup.out_features)
        self.fc_popup.weight.data.uniform_(-r, r)
        self.fc_popup.bias.data.fill_(0)

        r = np.sqrt(6.) / np.sqrt(self.fc_tool.in_features +
                                  self.fc_tool.out_features)
        self.fc_tool.weight.data.uniform_(-r, r)
        self.fc_tool.bias.data.fill_(0)


    def forward(self, videos):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized


        visual_feat = self.fc_visual(videos[:,:,:2048])
        popup_feat = self.fc_popup(videos[:,:,2048:4096])
        tool_feat = self.fc_tool(videos[:,:,4096:4135])


        # GCN visual
        visual_feat = visual_feat.permute(0, 2, 1)
        visual_feat = self.GCN_visual(visual_feat)
        visual_feat = visual_feat.permute(0, 2, 1)

        features_v = torch.mean(visual_feat,dim=1)
        features_v = torch.unsqueeze(features_v,1)

        # GCN popup
        popup_feat = popup_feat.permute(0, 2, 1)
        popup_feat = self.GCN_popup(popup_feat)
        popup_feat = popup_feat.permute(0, 2, 1)

        features_popup = torch.mean(popup_feat,dim=1)
        features_popup = torch.unsqueeze(features_popup,1)

        # tool feat
        features_tool = torch.mean(tool_feat,dim=1)
        features_tool = torch.unsqueeze(features_tool,1)

        # cat three feature
        features = torch.cat((features_v, features_popup, features_tool),1)

        # GCN three feat
        features = features.permute(0, 2, 1)
        features = self.GCN_feat(features)
        features = features.permute(0, 2, 1)

        # Attention
        feature_view = features.contiguous().view(features.size(0),-1)
        attn = self.feat_attn(feature_view)
        features = self.weighted_feature(features, attn)

        #features = self.bn(features)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class VCR(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    opt.finetune, opt.cnn_type,
                                    use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_abs=opt.use_abs)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        if opt.finetune:
            params += list(self.img_enc.cnn.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, videos, captions, lengths, volatile=False):
        """Compute the video and caption embeddings
        """
        # Set mini-batch dataset
        videos = Variable(videos, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            videos = videos.cuda()
            captions = captions.cuda()

        # Forward
        vid_emb = self.img_enc(videos)
        cap_emb = self.txt_enc(captions, lengths)
        return vid_emb, cap_emb

    def forward_loss(self, vid_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(vid_emb, cap_emb)
        self.logger.update('Le', loss.data[0], vid_emb.size(0))
        return loss

    def train_emb(self, videos, captions, lengths, ids=None, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])


        # compute the embeddings
        vid_emb, cap_emb = self.forward_emb(videos, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(vid_emb, cap_emb)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
