import torch
from vocab import Vocabulary
import evaluation_models


#evaluation_models.evalrank("pretrain_model/model_1/model_best.pth.tar","pretrain_model/model_2/model_best.pth.tar", data_path='../vsepp-master/data_SCAN/', split="test", fold5=False)
evaluation_models.evalrank("pretrain_model/model_1/model_best.pth.tar","pretrain_model/model_2/model_best.pth.tar", data_path='../matching_data/', split="test", fold5=False)