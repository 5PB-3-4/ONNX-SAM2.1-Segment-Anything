import argparse
import os
import tkinter as tk
from tkinter import filedialog
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from sam2 import SAM21A
from util import *


def esc_slash(src: str) -> str:
  dst = src.replace("\\", "/")
  return dst


class SAM21AnnotationApp:
  def __init__(self, root, sam2: SAM21A, image: np.ndarray):
    self.root = root
    self.sam2 = sam2

    self.image = None
    self.tk_image = None
    self.selected_label = None
    self.points = []
    self.rectangles = []
    self.label_ids = []
    self.label_colors = {}
    self.left_click_press_pos = None
    self.current_rectangle = None
    self.is_left_drag = False

    self.init_canvas(image)

    self.add_label(0)

  def init_canvas(self, image: np.ndarray):
    self.root.title("Image Annotation App")
    self.root.geometry("1500x900")

    # Image and canvas initialization
    self.canvas = tk.Canvas(self.root, width=1280, height=720)
    self.canvas.pack(side=tk.LEFT)

    # Browse button
    self.browse_button = tk.Button(self.root, text="Browse Image", command=self.browse_image)
    self.browse_button.pack()

    # Save button
    self.save_button = tk.Button(self.root, text="Save Image", command=self.save_image)
    self.save_button.pack()

    # Sidebar for labels
    self.label_frame = tk.Frame(self.root)
    self.label_frame.pack(side=tk.RIGHT, fill=tk.Y)

    self.label_listbox = tk.Listbox(self.label_frame)
    self.label_listbox.pack()

    self.add_label_button = tk.Button(self.label_frame, text="Add Label", command=self.add_label)
    self.add_label_button.pack()

    self.remove_label_button = tk.Button(self.label_frame, text="Remove Label", command=self.remove_label)
    self.remove_label_button.pack()
    # self.add_label_button.place()

    self.canvas.bind("<ButtonPress-1>", self.on_left_click_press)
    self.canvas.bind("<B1-Motion>", self.on_left_click_drag)
    self.canvas.bind("<ButtonRelease-1>", self.on_left_click_release)
    self.canvas.bind("<Button-3>", self.on_negative_point)

    self.image = image
    self.mask_image = self.image.copy()
    self.sam2.set_image(self.image)
    self.display_image()

  def browse_image(self):
    if self.image is not None:
      self.reset()

    file_path = filedialog.askopenfilename()
    if file_path:
      self.image_path = file_path
      self.image = cv2.imread(self.image_path)
      self.mask_image = self.image.copy()
      self.sam2.set_image(self.image)
      self.display_image()

  def save_image(self):
    if self.image is not None:
      self.reset()

    file_path = filedialog.asksaveasfilename()
    if file_path:
      file_path = esc_slash(file_path)
      folder_path = os.path.dirname(file_path)
      file_name = os.path.splitext(os.path.basename(file_path))[0]
      masks = self.sam2.get_masks()
      for i, mask in masks.items():
        img_path = os.path.join(folder_path, f"{file_name}_{i}.png")
        cv2.imwrite(img_path, mask)
    self.display_image()

  def display_image(self):
    if self.mask_image.shape[0] == 0:
      return

    # Convert the image to RGB (from BGR)
    rgb_image = cv2.cvtColor(self.mask_image, cv2.COLOR_BGR2RGB)
    # Convert the image to PIL format
    pil_image = Image.fromarray(rgb_image)
    self.tk_image = ImageTk.PhotoImage(image=pil_image)
    self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
    self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
    self.draw_points()
    self.draw_boxes()

  def add_label(self, label_id: int | None = None):
    if label_id is None:
      max_label = max(self.label_ids) if self.label_ids else 0

      # If the number of labels is less than the maximum label, use the next available label, otherwise use the next number
      if len(self.label_ids) == 0:
        label_id = 0
      elif len(self.label_ids) <= max_label:
        label_id = next(i for i in range(max_label + 1) if i not in self.label_ids)

      else:
        label_id = max_label + 1

    label = f"Label {label_id}"

    self.label_listbox.insert(tk.END, label)
    self.label_listbox.bind("<<ListboxSelect>>", self.select_label)
    self.label_listbox.selection_clear(0, tk.END)
    self.label_listbox.selection_set(tk.END)
    self.selected_label = label
    self.label_ids.append(label_id)

    b, g, r = colors[label_id]
    color = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    self.label_colors[label] = color

    # Set the background color of the listbox item
    self.label_listbox.itemconfig(tk.END, {"bg": color, "fg": "white"})

    # Sort the labels in the Listbox
    self.sort_labels()

  def select_label(self, event):
    widget = event.widget
    selection = widget.curselection()
    if selection:
      self.selected_label = widget.get(selection[0])

  def remove_label(self):
    selection = self.label_listbox.curselection()
    if selection:
      label = self.label_listbox.get(selection[0])
      self.label_listbox.delete(selection[0])

      label_id = int(label.split()[-1])
      self.label_ids.remove(label_id)

      # Remove points associated with this label
      points_to_remove = [point for point in self.points if point[3] == label]
      for point in points_to_remove:
        self.sam2.remove_point(np.array([point[1], point[2]]), label_id)
        self.canvas.delete(point[0])
        self.points.remove(point)

      # Remove boxes associated with this label
      boxes_to_remove = [box for box in self.rectangles if box[5] == label]
      for box in boxes_to_remove:
        self.sam2.remove_box(label_id)
        self.canvas.delete(box[0])
        self.rectangles.remove(box)

      masks = self.sam2.get_masks()
      self.mask_image = draw_masks(self.image, masks)
      self.display_image()

      self.selected_label = None
      if self.label_listbox.size() > 0:
        self.label_listbox.selection_set(0)
        self.selected_label = self.label_listbox.get(0)

      # Sort the labels in the Listbox
      self.sort_labels()

  def sort_labels(self):
    labels = list(self.label_listbox.get(0, tk.END))
    labels.sort(key=lambda x: int(x.split()[-1]))
    self.label_listbox.delete(0, tk.END)
    for label in labels:
      self.label_listbox.insert(tk.END, label)
      # Reapply the background color
      self.label_listbox.itemconfig(tk.END, {"bg": self.label_colors[label], "fg": "white"})

  def on_left_click_press(self, event):
    if self.image is None or not self.selected_label:
      return

    self.left_click_press_pos = (event.x, event.y)
    self.current_rectangle = self.canvas.create_rectangle(
      event.x,
      event.y,
      event.x,
      event.y,
      outline=self.label_colors[self.selected_label],
      width=2,
    )

  def on_left_click_drag(self, event):
    if self.image is None or not self.selected_label:
      return

    self.is_left_drag = True
    # Draw a rectangle while dragging
    x0, y0 = self.left_click_press_pos
    x1, y1 = event.x, event.y
    self.canvas.coords(self.current_rectangle, x0, y0, x1, y1)

  def on_left_click_release(self, event):
    if self.image is None or not self.selected_label:
      return

    distance = ((event.x - self.left_click_press_pos[0]) ** 2 + (event.y - self.left_click_press_pos[1]) ** 2) ** 0.5
    if self.is_left_drag and distance > 20:
      self.add_box(event)
      self.is_left_drag = False
    else:
      is_close, closest_point = self.is_close_to_point(event)
      is_close_to_rectangle, closest_rectangle = self.is_close_to_rectangle(event)
      if is_close:
        self.delete_point(closest_point)
      elif is_close_to_rectangle:
        self.delete_box(closest_rectangle)
      else:
        self.add_point(event, True)

    masks = self.sam2.get_masks()
    self.mask_image = draw_masks(self.image, masks)
    self.display_image()

  def on_negative_point(self, event):
    if self.image is None or not self.selected_label:
      return

    x, y = event.x, event.y
    label_id = int(self.selected_label.split()[-1])

    b, g, r = colors[label_id]
    color = f"#{255:02x}{0:02x}{0:02x}"

    radius = 3
    point = self.canvas.create_rectangle(x - radius * 3, y - radius, x + radius * 3, y + radius, fill=color, outline=color)
    self.points.append((point, x, y, self.selected_label, False))

    self.sam2.add_point(np.array([x, y]), np.array([False]), label_id)
    masks = self.sam2.get_masks()
    self.mask_image = draw_masks(self.image, masks)
    self.display_image()

    print(f"Added negative point at ({x}, {y}) with label '{self.selected_label}'")

  def add_point(self, event, is_positive=True):
    x, y = event.x, event.y
    label_id = int(self.selected_label.split()[-1])

    color = f"#{0:02x}{255:02x}{0:02x}" if is_positive else f"#{255:02x}{0:02x}{0:02x}"
    radius = 4
    point = self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)
    self.points.append((point, x, y, self.selected_label, is_positive))

    self.sam2.add_point(np.array([x, y]), np.array([True]), label_id)
    print(f"Added point at ({x}, {y}) with label '{self.selected_label}' and {'positive' if is_positive else 'negative'}")

  def add_box(self, event):
    x0, y0 = self.left_click_press_pos
    x1, y1 = event.x, event.y

    # Convert to top-left and bottom-right coordinates if not already
    if x0 > x1:
      x0, x1 = x1, x0
    if y0 > y1:
      y0, y1 = y1, y0

    label_id = int(self.selected_label.split()[-1])

    color = self.label_colors[self.selected_label]
    rectangle = self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2)

    # if labelid already has a rectangle, overwrite it
    if any(self.selected_label == rect[5] for rect in self.rectangles):
      for rect in self.rectangles:
        if self.selected_label == rect[5]:
          self.rectangles.remove(rect)

    self.rectangles.append((rectangle, x0, y0, x1, y1, self.selected_label))
    self.sam2.set_box(np.array([[x0, y0], [x1, y1]]), label_id)
    print(f"Added box at ({x0}, {y0}) to ({x1}, {y1}) with label '{self.selected_label}'")

  def delete_point(self, closest_point):
    self.canvas.delete(closest_point[0])
    self.points.remove(closest_point)
    label_id = int(closest_point[3].split()[-1])

    self.sam2.remove_point(np.array([closest_point[1], closest_point[2]]), label_id)

  def delete_box(self, closest_rectangle):
    print("Deleting box with label", closest_rectangle[5])
    self.canvas.delete(closest_rectangle[0])
    self.rectangles.remove(closest_rectangle)
    label_id = int(closest_rectangle[5].split()[-1])

    self.sam2.remove_box(label_id)

  def draw_points(self):
    for point in self.points:
      _, x, y, label, is_valid = point

      radius = 4
      if is_valid:
        color = f"#{0:02x}{255:02x}{0:02x}"
      else:
        color = f"#{255:02x}{0:02x}{0:02x}"
      self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)

  def draw_boxes(self):
    for rectangle in self.rectangles:
      _, x0, y0, x1, y1, label = rectangle
      color = self.label_colors[label]
      radius = 4
      self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2)
      self.canvas.create_oval(x0 - radius, y0 - radius, x0 + radius, y0 + radius, fill=color, outline=color)
      self.canvas.create_oval(x1 - radius, y1 - radius, x1 + radius, y1 + radius, fill=color, outline=color)

  def generate_color(self):
    import random

    def r():
      return random.randint(0, 255)

    return f"#{r():02x}{r():02x}{r():02x}"

  def reset(self):
    self.image = None
    self.mask_image = None
    self.tk_image = None
    self.canvas.delete("all")
    self.points = []
    self.rectangles = []
    self.label_listbox.delete(0, tk.END)
    self.label_ids = []
    self.selected_label = None
    self.add_label(0)

  def is_close_to_point(self, event, distance_thres=10):
    # Check if the point is close to an existing point for deletion
    x, y = event.x, event.y
    closest_point = None
    closest_distance = float("inf")

    for point in self.points:
      _, px, py, _, _ = point
      distance = (x - px) ** 2 + (y - py) ** 2
      if distance < closest_distance:
        closest_distance = distance
        closest_point = point

    is_close = closest_distance < distance_thres**2

    return is_close, closest_point

  def is_close_to_rectangle(self, event, distance_thres=10):
    # Check if the point is close to an existing point for deletion
    x, y = event.x, event.y
    closest_rectangle = None
    closest_distance = float("inf")

    for rectangle in self.rectangles:
      _, x0, y0, x1, y1, _ = rectangle
      distance = (x - x0) ** 2 + (y - y0) ** 2
      if distance < closest_distance:
        closest_distance = distance
        closest_rectangle = rectangle

      distance = (x - x1) ** 2 + (y - y1) ** 2
      if distance < closest_distance:
        closest_distance = distance
        closest_rectangle = rectangle

    is_close = closest_distance < distance_thres**2

    return is_close, closest_rectangle


def app_main(image: np.ndarray, args: Any) -> None:
  root = tk.Tk()

  model_path = {
    "Encoder": args.encode,
    "Decoder": args.decode,
  }
  sam2 = SAM21A(model_path)

  _app = SAM21AnnotationApp(root, sam2, image)
  root.mainloop()


def script_main(image: np.ndarray, args: Any) -> None:
  model_path = {
    "Encoder": args.encode,
    "Decoder": args.decode,
  }

  # Initialize models
  sam2 = SAM21A(model_path)

  # Set image
  sam2.set_image(image)
  print("image size: ", image.shape)

  # Add points
  point_coords = [np.array([100, 440]), np.array([[360, 275], [370, 210]]), np.array([810, 0]), np.array([920, 514])]
  point_labels = [np.array([1]), np.array([1, 1]), np.array([1]), np.array([1])]

  for label_id, (point_coord, point_label) in enumerate(zip(point_coords, point_labels, strict=False)):
    # print("label: ", point_label)
    sam2.add_point(point_coord, point_label, label_id)

    masks = sam2.get_masks()

    # Draw masks
    masked_img = draw_masks(img, masks)
    # masked_img = masks[label_id]

    # masks = masks[0]
    # scores = sam2.scores_
    # nidx = np.argmin(scores)

    # maskl = []
    # for i in range(4):
    #   if i!= int(nidx):
    #     maskl.append(masks[0][i])
    # masked_img = np.array(maskl)
    # masked_img = np.transpose(masked_img, (1, 2, 0))

    # mask00 = cv2.resize(masks[0][0], None, None, 0.5, 0.5, 1)
    # mask01 = cv2.resize(masks[0][1], None, None, 0.5, 0.5, 1)
    # mask10 = cv2.resize(masks[0][2], None, None, 0.5, 0.5, 1)
    # mask11 = cv2.resize(masks[0][3], None, None, 0.5, 0.5, 1)

    # mask0 = np.hstack((mask00, mask01))
    # mask1 = np.hstack((mask10, mask11))
    # masked_img = np.vstack((mask0, mask1))

    # masked_img = masked_img[0] #np.transpose(masked_img[0], (1, 2, 0))
    # print(masked_img.shape)
    # masked_img = cv2.normalize(masked_img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    # _, masked_img = cv2.threshold(masked_img, 1.0, 255.0, cv2.THRESH_BINARY)

    # masked_img = cv2.morphologyEx(masked_img, cv2.MORPH_CROSS, np.ones((3,3),np.uint8))

    cv2.imshow("masked_img", masked_img)

    if cv2.waitKey(4000) & 0xFF == ord("q"):
      break

  cv2.waitKey(0)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="SAM2 Demo")
  parser.add_argument("--input", "-i", type=str, help="Set input path to a certain image.")
  parser.add_argument("--encode", "-e", type=str, help="Set model path to encoder (preprocess).")
  parser.add_argument("--decode", "-d", type=str, help="Set input path to a decoder (main process).")
  parser.add_argument("--appmode", "-a", action="store_true", help="Set flag to entry annotation app.")
  args = parser.parse_args()

  try:
    img = cv2.imread(esc_slash(str(args.input)), cv2.IMREAD_COLOR)
    if img is not None:
      if args.appmode:
        app_main(img, args)
      else:
        script_main(img, args)
  except Exception:
    pass
