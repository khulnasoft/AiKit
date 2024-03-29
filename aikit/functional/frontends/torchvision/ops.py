import aikit
from aikit.functional.frontends.torch.func_wrapper import to_aikit_arrays_and_back
from aikit.func_wrapper import with_supported_dtypes, with_unsupported_device_and_dtypes


@to_aikit_arrays_and_back
def batched_nms(boxes, scores, idxs, iou_threshold):
    if boxes.size == 0:
        return aikit.array([], dtype=aikit.int64)
    else:
        max_coordinate = boxes.max()
        boxes_dtype = boxes.dtype
        offsets = idxs.astype(boxes_dtype) * (
            max_coordinate + aikit.array(1, dtype=boxes_dtype)
        )
        boxes_for_nms = boxes + offsets[:, None]
        keep = nms(boxes_for_nms, scores, iou_threshold)
        return keep


@to_aikit_arrays_and_back
def box_area(boxes):
    return aikit.prod(boxes[..., 2:] - boxes[..., :2], axis=-1)


@to_aikit_arrays_and_back
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = aikit.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = aikit.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clip(x_min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


@with_unsupported_device_and_dtypes(
    {
        "2.1.2 and below": {
            "cpu": ("float16",),
        }
    },
    "torch",
)
@to_aikit_arrays_and_back
def clip_boxes_to_image(boxes, size):
    height, width = size
    boxes_x = boxes[..., 0::2].clip(0, width)
    boxes_y = boxes[..., 1::2].clip(0, height)
    clipped_boxes = aikit.stack([boxes_x, boxes_y], axis=-1)
    return clipped_boxes.reshape(boxes.shape).astype(boxes.dtype)


@to_aikit_arrays_and_back
def nms(boxes, scores, iou_threshold):
    return aikit.nms(boxes, scores, iou_threshold)


@to_aikit_arrays_and_back
def remove_small_boxes(boxes, min_size):
    w, h = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]
    return aikit.nonzero((w >= min_size) & (h >= min_size))[0]


@with_supported_dtypes({"2.1.2 and below": ("float32", "float64")}, "torch")
@to_aikit_arrays_and_back
def roi_align(
    input, boxes, output_size, spatial_scale=1.0, sampling_ratio=1, aligned=False
):
    return aikit.roi_align(
        input, boxes, output_size, spatial_scale, sampling_ratio, aligned
    )
