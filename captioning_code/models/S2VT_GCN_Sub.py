import torch
import torch.nn as nn
import numpy as np
from .Res_GCN import Res_GCN

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class S2VT_GCN_Sub(nn.Module):
    def __init__(self, encoder, decoder, embed_size=2048):
        """

        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """
        super(S2VT_GCN_Sub, self).__init__()
        self.encoder = encoder
        self.decoder = decoder



        self.fc_visual = nn.Linear(2048, embed_size)
        self.fc_popup = nn.Linear(2048, embed_size)
        self.fc_tool = nn.Linear(39, embed_size)

        self.init_weights()

        self.GCN_visual = Res_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.GCN_popup = Res_GCN(in_channels=embed_size, inter_channels=embed_size)

        self.GCN_feat = Res_GCN(in_channels=embed_size, inter_channels=embed_size)

        # self.feat_attn = nn.Linear(embed_size*3, 3)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.feat_attn =nn.Sequential(
            nn.Linear(3, 3),
            nn.Sigmoid()
            )



    def weighted_feature(self, feat, attn):
        attn = torch.unsqueeze(attn, 2)

        attn_feat = feat[:,:,:,0]

        for i in range(feat.shape[1]):
            cur_feat = torch.matmul(feat[:,i,:,:], attn)  
            cur_feat = torch.squeeze(cur_feat)
            attn_feat[:,i,:] = cur_feat


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

    def forward(self, vid_feats, target_variable=None,
                mode='train', opt={}):
        """

        Args:
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): groung truth labels

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """


        visual_feat = self.fc_visual(vid_feats[:,:,:2048])
        popup_feat = self.fc_popup(vid_feats[:,:,2048:4096])
        tool_feat = self.fc_tool(vid_feats[:,:,4096:4135])


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

        #attention

        feature_view = self.avg_pool(features)
        feature_view = feature_view.contiguous().view(feature_view.size(0),-1)
        attn = self.feat_attn(feature_view)


        # vid_feats = self.weighted_feature(features, attn)
        visual_feat = torch.unsqueeze(visual_feat,3)
        popup_feat = torch.unsqueeze(popup_feat,3)
        tool_feat = torch.unsqueeze(tool_feat,3)
        squ_features = features = torch.cat((visual_feat, popup_feat, tool_feat),3)

        vid_feats = self.weighted_feature(squ_features, attn)

        # vid_feats = l2norm(vid_feats)
 

        encoder_outputs, encoder_hidden = self.encoder(vid_feats)
        seq_prob, seq_preds = self.decoder(encoder_outputs, encoder_hidden, target_variable, mode, opt)
        return seq_prob, seq_preds
