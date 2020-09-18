from vocab import Vocabulary
import evaluation
# import evaluation_vsepp
# evaluation_vsepp.evalrank("runs/save_ori_vse/coco_vse++_combine/model_best.pth.tar", data_path='data/', split="test", fold5=False) 


# evaluation.evalrank("runs/coco_vse++_combine_P_S/model_best.pth.tar", data_path='data/', split="test", fold5=False) 

# for i in range(1,17):
# 	print('model:' + str(i) + '\n')
# 	evaluation.evalrank("runs/save_fc_attn/coco_combine_double_GCN_attn_" + str(i) +"/model_best.pth.tar", data_path='../vsepp-master/data_SCAN/', split="test", fold5=False) 
# 	print('\n')
# 	# print('\n')
# 	# evaluation.evalrank("runs/coco_combine_double_GCN_attn/checkpoint.pth.tar", data_path='../vsepp-master/data_SCAN/', split="test", fold5=False) 



# for i in range(1,10):
# 	print('model:' + str(i) + '\n')
# 	evaluation.evalrank("runs/camera/GCN_Attn_" + str(i) +"/model_best.pth.tar", data_path='../vsepp-master/data_SCAN/', split="test", fold5=False) 
# 	print('\n')
# 	# print('\n')
# 	# evaluation.evalrank("runs/coco_combine_double_GCN_attn/checkpoint.pth.tar", data_path='../vsepp-master/data_SCAN/', split="test", fold5=False) 


for i in range(1,5):
	print('model:' + str(i) + '\n')
	evaluation.evalrank("runs/camera/Only_Visual_GCN_Attn_" + str(i) +"/model_best.pth.tar", data_path='../vsepp-master/data_SCAN/', split="test", fold5=False) 
	print('\n')
	# print('\n')
	# evaluation.evalrank("runs/coco_combine_double_GCN_attn/checkpoint.pth.tar", data_path='../vsepp-master/data_SCAN/', split="test", fold5=False) 

