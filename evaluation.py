"""
CUDA_VISIBLE_DEVICES=1 \
python demo/test_ap_on_coco.py \
 -c groundingdino/config/GroundingDINO_SwinB_cfg.py \
 -p ./weights/groundingdino_swinb_cogcoor.pth \
 --anno_path ./data/test/_annotations.coco.json \
 --image_dir ./data/test
"""
"""

CUDA_VISIBLE_DEVICES=0 python demo/inference_on_a_image.py \
-c groundingdino/config/GroundingDINO_SwinB_cfg.py \
-p ./weights/groundingdino_swinb_cogcoor.pth \
-i .asset/Image_20191007024433999.jpg \
-o logs/1111 \
-t "The small round mouth in the center of the top of the glass bottle"
"""