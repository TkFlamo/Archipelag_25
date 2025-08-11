def bbox_intersection(bbox_1, bbox_2):
    m_w = 0
    m_h = 0
    if (bbox_1[0] < bbox_2[0]):
        m_w = max(min(bbox_1[0] + bbox_1[2] - bbox_2[0], bbox_2[2]), 0)
    else:
        m_w = max(min(bbox_2[0] + bbox_2[2] - bbox_1[0], bbox_1[2]), 0)
    if (bbox_1[1] < bbox_2[1]):
        m_h = max(min(bbox_1[1] + bbox_1[3] - bbox_2[1], bbox_2[3]), 0)
    else:
        m_h = max(min(bbox_2[1] + bbox_2[3] - bbox_1[1], bbox_1[3]), 0)
    intersection = m_w * m_h
    return intersection

def ios(bbox_1, bbox_2):
    intersection = bbox_intersection(bbox_1, bbox_2)
    smaller_bbox_area = min(bbox_1[2] * bbox_1[3], bbox_2[2] * bbox_2[3])

    if intersection > smaller_bbox_area:
        print("Error in ios function")
        print(intersection, smaller_bbox_area)
        
    if smaller_bbox_area > 0:
        ios = intersection / smaller_bbox_area
    else:
        ios = 0
    return ios