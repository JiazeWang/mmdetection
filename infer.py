from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = 'configs/mask_rcnn_r50_fpn_1x.py'
checkpoint_file = 'checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'img_00001.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
#show_result(img, result, model.CLASSES)
# or save the visualization results to image files
print(model.CLASSES)
show_result(img, result, model.CLASSES, out_file='result_00001.jpg')
