import time
from typing import Any, Union

import cv2
import numpy as np
import onnxruntime as ort


class SAM21A:
  def __init__(self, model_path=None, device=None):
    # infer setting
    self.encoder = self.load_model(model_path["Encoder"], device)
    self.decoder = self.load_model(model_path["Decoder"], device)

    # decoder parameters
    self.encode_results = ()
    self.tensor_size = (0, 0)
    self.image_size = (0, 0)
    self.scale_factor = 4
    self.mask_threshold = 0.0

    # user input data
    self.decode_insts = {}
    self.point_coords = {}
    self.box_coords = {}
    self.point_labels = {}
    self.masks = {}
    self.scores = {}

  def set_image(self, image: np.ndarray) -> None:
    # encode image
    tensor = self.preprocess(image)
    self.encode_results = self.infer(self.encoder, tensor)

    # reset parameters
    self.decode_settings = {}
    self.point_coords = {}
    self.box_coords = {}
    self.point_labels = {}
    self.masks = {}
    self.scores = {}

  def preprocess(self, image: np.ndarray) -> np.ndarray:
    # set input tensor shape for encoder
    self.tensor_size = tuple((self.encoder["input_shapes"][0])[::-1][:2])

    # opencv operation
    image_size = list(image.shape)[:2][::-1]
    self.image_size = tuple(image_size)
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, self.tensor_size)

    # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_img = (input_img / 255.0 - mean) / std
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
    return input_tensor

  def add_point(self, point_coords: np.ndarray, is_positive: np.ndarray, label_id: int) -> dict[int, np.ndarray]:
    if label_id not in self.decode_insts:
      self.decode_insts[label_id] = self.decoder

    if label_id not in self.point_coords:
      self.point_coords[label_id] = np.expand_dims(point_coords.ravel(), 0)
      self.point_labels[label_id] = np.expand_dims(is_positive.ravel().astype(np.int32), 0)
    else:
      self.point_coords[label_id] = np.append(self.point_coords[label_id].ravel(), point_coords.ravel())
      self.point_labels[label_id] = np.append(self.point_labels[label_id].ravel(), is_positive.ravel())
      self.point_labels[label_id] = np.expand_dims(self.point_labels[label_id].ravel(), 0)
    self.point_coords[label_id] = self.point_coords[label_id].reshape((-1, 2))

    return self.decode_mask(label_id)

  def set_box(self, box_coords: np.ndarray, label_id: int) -> dict[int, np.ndarray]:
    if label_id not in self.decode_insts:
      self.decode_insts[label_id] = self.decoder

    if box_coords.size == 4:  # Convert from 1x4 to 2x2
      box_coords = box_coords.reshape((-1, 2))

    self.box_coords[label_id] = box_coords

    return self.decode_mask(label_id)

  def remove_point(self, point_coords: np.ndarray, label_id: int) -> dict[int, np.ndarray]:
    point_id = np.where(
      (self.point_coords[label_id][:, 0] == point_coords[0]) & (self.point_coords[label_id][:, 1] == point_coords[1]),
    )[0]
    if point_id.size:
      self.point_coords[label_id] = np.delete(self.point_coords[label_id], point_id[0], axis=0)
      self.point_labels[label_id] = np.delete(self.point_labels[label_id], point_id[0], axis=-1)

    return self.decode_mask(label_id)

  def remove_box(self, label_id: int) -> dict[int, np.ndarray]:
    del self.box_coords[label_id]
    return self.decode_mask(label_id)

  def decode_mask(self, label_id: int) -> dict[int, np.ndarray]:
    concat_coords, concat_labels = self.merge_points_and_boxes(label_id)
    point_coords, point_labels, mask_input, has_mask_input, original_size = self.set_input_decode(concat_coords, concat_labels)

    decoder = self.decode_insts[label_id]
    image_embed, high_res_feats_0, high_res_feats_1 = self.encode_results
    if concat_coords.size == 0:
      mask = np.zeros(self.image_size, dtype=np.uint8)
    else:
      results = self.infer(
        decoder,
        image_embed,
        high_res_feats_0,
        high_res_feats_1,
        point_coords,
        point_labels,
        mask_input,
        has_mask_input,
        original_size,
      )
      mask, _ = self.postprocess(results)
    self.masks[label_id] = mask

    return self.masks

  def merge_points_and_boxes(self, label_id: int) -> tuple[np.ndarray, np.ndarray]:
    concat_coords = []
    concat_labels = []
    has_points = label_id in self.point_coords
    has_boxes = label_id in self.box_coords

    if not has_points and not has_boxes:
      return np.empty(0), np.empty(0)

    if has_points:
      concat_coords.append(self.point_coords[label_id])
      concat_labels.append(self.point_labels[label_id].ravel())
    if has_boxes:
      concat_coords.append(self.box_coords[label_id])
      concat_labels.append(np.array([2, 3]))
    concat_coords = np.concatenate(concat_coords, axis=0)
    concat_labels = np.concatenate(concat_labels, axis=0)

    # concat_coords = np.reshape(concat_coords, (1, -1, 2))
    # concat_labels = np.reshape(concat_labels, (concat_coords.shape[0], concat_coords.shape[1]))

    return concat_coords, concat_labels

  def get_masks(self) -> dict[int, np.ndarray]:
    return self.masks

  def postprocess(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    scores = outputs[1].squeeze()
    masks = outputs[0]
    # print("value: ", np.unique(np.sort(np.unique(masks[0][0])).astype(np.int32)))

    masks_selected = np.zeros_like(masks[0][0])
    nidx = np.argmin(scores)
    for i in range(scores.size):
      if i != int(nidx) or scores[i] > 0.7:
        masks_selected += masks[0][i] > 0
    masks = np.array(masks_selected > 0, np.uint8)
    masks = cv2.convertScaleAbs(masks, None, 255.0, 0.0)

    # masks = masks * (masks > self.mask_threshold)
    # self.scores_ = scores
    # masks = masks.astype(np.uint8).squeeze()
    return masks, scores

  def set_input_decode(self, point_coords: np.ndarray, point_labels: np.ndarray):
    input_point_coords, input_point_labels = self.prepare_points(point_coords, point_labels)
    num_labels = input_point_labels.shape[0]
    mask_input = np.zeros(
      (num_labels, 1, int(self.tensor_size[0] / self.scale_factor), int(self.tensor_size[1] / self.scale_factor)),
      dtype=np.float32,
    )
    has_mask_input = np.array([0], dtype=np.float32)
    original_size = np.array(list(self.image_size), dtype=np.int64)[::-1]

    return input_point_coords, input_point_labels, mask_input, has_mask_input, original_size

  def prepare_points(self, point_coords: np.ndarray, point_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    input_point_coords = point_coords[np.newaxis, ...]
    input_point_labels = point_labels[np.newaxis, ...]

    input_point_coords[..., 0] = input_point_coords[..., 0] / self.image_size[0] * self.tensor_size[0]  # Normalize x
    input_point_coords[..., 1] = input_point_coords[..., 1] / self.image_size[1] * self.tensor_size[1]  # Normalize y

    return input_point_coords.astype(np.float32), input_point_labels.astype(np.float32)

  def infer(self, infer_data: Any, *inputs):
    input_data = {}
    for i, key in enumerate(infer_data["input_names"]):
      input_data.setdefault(key, inputs[i])

    start = time.perf_counter()
    results = infer_data["session"].run(infer_data["output_names"], input_data)
    print(f"infer time: {(time.perf_counter() - start) * 1000:.2f} ms")
    return results

  def load_model(self, model_path: str, providers=None) -> dict[str, list | ort.InferenceSession]:
    if providers is None:
      providers = ort.get_available_providers()  # ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    # set layar name/shape
    input_metadata = session.get_inputs()
    output_metadata = session.get_outputs()
    input_name_list = [input_metadata[i].name for i in range(len(input_metadata))]
    output_name_list = [output_metadata[i].name for i in range(len(output_metadata))]
    input_shape_list = [input_metadata[i].shape for i in range(len(input_metadata))]
    output_shape_list = [output_metadata[i].shape for i in range(len(output_metadata))]

    infer_data = {
      "session": session,
      "input_names": input_name_list,
      "output_names": output_name_list,
      "input_shapes": input_shape_list,
      "output_shapes": output_shape_list,
    }
    return infer_data
