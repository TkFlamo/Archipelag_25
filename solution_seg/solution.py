import numpy as np
from typing import List, Union

from ultralytics import YOLO

from patched_yolo_infer_v1_seg import (
    MakeCropsDetectThem,
    CombineDetections,
)

MODEL_PATH = 'seg_1024_2.pt'
model = YOLO(MODEL_PATH)

def infer_image_bbox(image_bgr: np.ndarray) -> List[dict]:
    height, width = image_bgr.shape[:2]
    
    element_crops = MakeCropsDetectThem(
        image=image_bgr,
        model=model,
        shape_x=1024,
        shape_y=1024,
        overlap_x=20,
        overlap_y=20,
        max_crops_x=10,
        max_crops_y=10,
        conf=0.22,
        iou=0.1,
        imgsz=1024,
        classes_list=[0],
    )
    
    result = CombineDetections(element_crops, nms_threshold=0.3, sorter_bins=5, class_agnostic_nms=False)
    bbox_list = result.filtered_boxes
    conf_list = result.filtered_confidences
    class_list = result.filtered_classes_id

    res_list = []
    for i in range(len(bbox_list)):
        bbox = bbox_list[i]
        class_id = class_list[i]
        score = conf_list[i]

        xc = (bbox[0] + bbox[2]) / 2 / width
        yc = (bbox[1] + bbox[3]) / 2 / height
        w = (bbox[2] - bbox[0]) / width
        h = (bbox[3] - bbox[1]) / height

        formatted = {
            'xc': xc,
            'yc': yc,
            'w': w,
            'h': h,
            'label': class_id,
            'score': score
        }
        res_list.append(formatted)
    return res_list


def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    """Функция производит инференс модели на одном или нескольких изображениях.

    Args:
        images (Union[List[np.ndarray], np.ndarray]): Список изображений или одно изображение.

    Returns:
        List[List[dict]]: Список списков словарей с результатами предикта 
        на найденных изображениях.
        Пример выходных данных:
        [
            [
                {
                    'xc': 0.5,
                    'yc': 0.5,
                    'w': 0.2,
                    'h': 0.3,
                    'label': 0,
                    'score': 0.95
                },
                ...
            ],
            ...
        ]
    """    
    results = []
    if isinstance(images, np.ndarray):
        images = [images]

    # Обрабатываем каждое изображение из полученного списка
    for image in images:    
        image_results = infer_image_bbox(image)
        results.append(image_results)
    
    return results
