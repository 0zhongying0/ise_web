import os 
from tensorflow.python import pywrap_tensorflow 

 
checkpoint_path = os.path.join('train_logs/cicids2017/alad_snFalse_dzzFalse/dzzenabledFalse/fm/label0/rd42','model.ckpt-0') # 保存的ckpt文件名，不一定是这个 
# Read data from checkpoint file 
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) 
var_to_shape_map = reader.get_variable_to_shape_map() 
# Print tensor name and values 
for key in var_to_shape_map: 
	print("tensor_name: ", key) 
	# print(reader.get_tensor(key)) # 打印变量的值，对我们查找问题没啥影响，打印出来反而影响找问题
