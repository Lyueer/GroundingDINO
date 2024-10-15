from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os

# 清除缓存
os.environ['TRANSFORMERS_CACHE'] = "./bert-base-uncased"

# model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
model = load_model("groundingdino/config/GroundingDINO_SwinB_cfg.py", "weights/groundingdino_swinb_cogcoor.pth")
IMAGE_PATH = ".asset/Image_20191006232054610.jpg"
# Image_20191007024433999.jpg
# Image_20191006232054610.jpg
# Image_20191007031043199.jpg
# Image_20191006232102720.jpg
TEXT_PROMPT = "The small round mouth in the center of the top of the glass bottle"
BOX_TRESHOLD = 0.25
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)

"""
CUDA_VISIBLE_DEVICES=0 python demo/inference_on_a_image.py \
-c groundingdino/config/GroundingDINO_SwinB_cfg.py \
-p ./weights/groundingdino_swinb_cogcoor.pth \
-i .asset/Image_20191007024433999.jpg \
-o logs/1111 \
-t "The small round mouth in the center of the top of the glass bottle" 
--token_spans "[[[0, 20],  [50, 65]]]"
"""
"""
CUDA_VISIBLE_DEVICES=1 python demo/inference_on_a_image.py \
-c groundingdino/config/GroundingDINO_SwinB_cfg.py \
-p ./weights/groundingdino_swinb_cogcoor.pth \
-i .asset/Image_20191006232102720.jpg \
-o logs/1111 \
-t "The round mouth in the lower center of the top of the water droplet shaped glass bottle" \
--box_threshold 0.25 \
--text_threshold 0.25 
"""
