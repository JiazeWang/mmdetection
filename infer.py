from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = 'configs/mask_rcnn_r50_fpn_1x4.py'
checkpoint_file = 'model/mask_rcnn_r50_c4_2x-3cf169a9.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'model/sample/img_00001.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
#show_result(img, result, model.CLASSES)
# or save the visualization results to image files
print(model.CLASSES)
show_result(img, result, model.CLASSES, out_file='model/result_00001.jpg')
