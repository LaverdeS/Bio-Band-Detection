# if __name__ == '__main__':
#     # for image in inputs:
#     #     read image
#     #     process image
#     #     print block infomration array
#     #     write output to output folder

"""Build Graphic Interface to control parameters of the program"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from os import walk, listdir
from os.path import isfile, join
import pandas as pd
import argparse
from tqdm import tqdm
import tkinter as tk
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from PIL import Image, ImageTk
from matplotlib import cm
import json


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def load_images(path):
    images = []
    for (dirpath, dirnames, filenames) in walk(path):
        for filename in filenames:
            if filename.endswith(".png"):
                images.append(load_image(path + filename))
    return images


def load_image_filenames(path):
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    return filenames


def apply_threshold(img, args):
    if args.threshold_type == "binary":
        img = cv2.threshold(img, args.threshold, 255, cv2.THRESH_BINARY)[1]
    elif args.threshold_type == "binary_inv":
        img = cv2.threshold(img, args.threshold, 255, cv2.THRESH_BINARY_INV)[1]
    elif args.threshold_type == "trunc":
        img = cv2.threshold(img, args.threshold, 255, cv2.THRESH_TRUNC)[1]
    elif args.threshold_type == "tozero":
        img = cv2.threshold(img, args.threshold, 255, cv2.THRESH_TOZERO)[1]
    elif args.threshold_type == "tozero_inv":
        img = cv2.threshold(img, args.threshold, 255, cv2.THRESH_TOZERO_INV)[1]
    return img


def get_contours(img, args):
    # get contours
    img = cv2.medianBlur(img, args.blur)
    img = cv2.Canny(img, args.canny, args.canny * 2)
    img = cv2.dilate(img, None, iterations=args.dilate)
    img = cv2.erode(img, None, iterations=args.erode)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # filter contours
    contours = filter_by_area(contours, args)
    contours = filter_by_ratio(contours, args)
    contours = filter_by_circularity(contours, args)
    # contours = filter_by_inertia_ratio(contours, args)
    contours = filter_by_color(contours, img, args)
    contours = filter_by_convexity(contours, args)
    contours = filter_by_min_dist_between_blobs(contours, args)
    contours = filter_by_pixel_value(contours, img, args)
    return contours


def get_contours_from_images(images, args):
    contours = []
    for img in images:
        contours.append(get_contours(img, args))
    return contours


def filter_by_area(contours, args):
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < args.min_area or area > args.max_area:
            continue
        filtered_contours.append(contour)
    return filtered_contours


def filter_by_ratio(contours, args):
    filtered_contours = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        ratio = 4 * np.pi * cv2.contourArea(contour) / perimeter ** 2
        if ratio < args.min_ratio or ratio > args.max_ratio:
            continue
        filtered_contours.append(contour)
    return filtered_contours


def filter_by_pixel_value(contours, img, args):
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if np.mean(img[y:y + h, x:x + w]) < args.min_pixel_value:
            continue
        filtered_contours.append(contour)
    return filtered_contours


def filter_by_convexity(contours, args):
    filtered_contours = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        convexity = float(cv2.contourArea(contour)) / hull_area
        if convexity < args.min_convexity or convexity > args.max_convexity:
            continue
        filtered_contours.append(contour)
    return filtered_contours


def filter_by_color(contours, img, args):
    filtered_contours = []
    for contour in contours:
        # filter using min_color and max_color
        x, y, w, h = cv2.boundingRect(contour)
        if np.mean(img[y:y + h, x:x + w]) < args.min_color or np.mean(img[y:y + h, x:x + w]) > args.max_color:
            continue
        filtered_contours.append(contour)
    return filtered_contours


def filter_by_inertia_ratio(contours, args):
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area == 0:
            continue
        inertia_ratio = cv2.matchShapes(contour, cv2.minAreaRect(contour)[0], 1, 0.0)
        if inertia_ratio < args.min_inertia_ratio or inertia_ratio > args.max_inertia_ratio:
            continue
        filtered_contours.append(contour)
    return filtered_contours


def filter_contours(contours, img, args):
    filtered_contours = []
    for contour in contours:
        if args.filter_by_area:
            area = cv2.contourArea(contour)
            if area < args.min_area or area > args.max_area:
                continue
        if args.filter_by_ratio:
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            ratio = 4 * np.pi * cv2.contourArea(contour) / perimeter ** 2
            if ratio < args.min_ratio or ratio > args.max_ratio:
                continue
        if args.filter_by_pixel_value:
            x, y, w, h = cv2.boundingRect(contour)
            if np.mean(img[y:y + h, x:x + w]) < args.min_pixel_value:
                continue
        if args.filter_by_convexity:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            convexity = float(cv2.contourArea(contour)) / hull_area
            if convexity < args.min_convexity or convexity > args.max_convexity:
                continue
        if args.filter_by_color:
            x, y, w, h = cv2.boundingRect(contour)
            if np.mean(img[y:y + h, x:x + w]) < args.min_color:
                continue
        if args.filter_by_circularity:
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            ratio = 4 * np.pi * cv2.contourArea(contour) / perimeter ** 2
            if ratio < args.min_circularity or ratio > args.max_circularity:
                continue
        filtered_contours.append(contour)
    return filtered_contours


def filter_by_circularity(contours, args):
    filtered_contours = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        ratio = 4 * np.pi * cv2.contourArea(contour) / perimeter ** 2
        if ratio < args.min_circularity or ratio > args.max_circularity:
            continue
        filtered_contours.append(contour)
    return filtered_contours


def filter_by_inertia(contours, args):
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area == 0:
            continue
        inertia = cv2.matchShapes(contour, np.array([[[0, 0]], [[0, 1]], [[1, 1]], [[1, 0]]]), cv2.CONTOURS_MATCH_I1, 0)
        if inertia < args.min_inertia or inertia > args.max_inertia:
            continue
        filtered_contours.append(contour)
    return filtered_contours


def filter_by_min_dist_between_blobs(contours, args):
    filtered_contours = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        for j, contour2 in enumerate(contours):
            if i == j:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(contour2)
            if abs(x - x2) < args.min_dist_between_blobs and abs(y - y2) < args.min_dist_between_blobs:
                break
        else:
            filtered_contours.append(contour)
    return filtered_contours


def plot_red_contours(contours, img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blur", type=int, default=1)
    parser.add_argument("--canny", type=int, default=33)
    parser.add_argument("--dilate", type=int, default=5)
    parser.add_argument("--erode", type=int, default=0)

    parser.add_argument("--min_area", type=int, default=888)
    parser.add_argument("--max_area", type=int, default=10000)

    parser.add_argument("--min_ratio", type=float, default=0.0)
    parser.add_argument("--max_ratio", type=float, default=5)

    parser.add_argument("--min_circularity", type=float, default=0.0)
    parser.add_argument("--max_circularity", type=float, default=1.0)

    # parser.add_argument("--min_inertia_ratio", type=float, default=0.5)
    # parser.add_argument("--max_inertia_ratio", type=float, default=1.5)

    parser.add_argument("--min_convexity", type=float, default=0.0)
    parser.add_argument("--max_convexity", type=float, default=1.0)

    parser.add_argument("--min_color", type=float, default=0)
    parser.add_argument("--max_color", type=float, default=255)

    parser.add_argument("--min_pixel_value", type=float, default=0.0)
    parser.add_argument("--max_pixel_value", type=float, default=255.0)

    parser.add_argument("--min_dist_between_blobs", type=int, default=10)

    parser.add_argument("--threshold", type=int, default=127)
    args = parser.parse_args()
    return args


# control a set of parameters with sliders

class myGUI(object):
    def __init__(self, path, filenames, args):
        self.path = path
        self.args = args
        self.root = tk.Tk()
        self.canvas = tk.Canvas(width=1250, height=80, bg='black')
        self.canvas.pack()

        self.filenames = [path + f for f in filenames]
        self.index = 0
        # self.image = Image.open(self.filenames[self.index])
        self.contours = get_contours_from_images(load_images(path), args)

        # image from numpy array
        self.image = cv2.imread(self.filenames[self.index], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(img, self.contours[self.index], -1, (0, 0, 255), 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(img_rgb)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # create a frame for the sliders
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=1)

        # threshold slider
        # self.threshold = tk.Scale(self.frame, from_=0, to=255, orient=tk.HORIZONTAL, label='threshold')
        # self.threshold.set(args.threshold)
        # self.threshold.pack()

        # create the sliders
        self.blur = tk.Scale(self.frame, from_=1, to=20, orient=tk.HORIZONTAL, label="blur", command=self.update_image, length=600)
        self.blur.set(args.blur)
        self.blur.pack()

        self.canny = tk.Scale(self.frame, from_=1, to=100, orient=tk.HORIZONTAL, label="canny",
                              command=self.update_image, length=600)
        self.canny.set(args.canny)
        self.canny.pack()

        self.dilate = tk.Scale(self.frame, from_=1, to=10, orient=tk.HORIZONTAL, label="dilate",
                               command=self.update_image, length=600)
        self.dilate.set(args.dilate)
        self.dilate.pack()

        self.erode = tk.Scale(self.frame, from_=0, to=10, orient=tk.HORIZONTAL, label="erode",
                              command=self.update_image, length=600)
        self.erode.set(args.erode)
        self.erode.pack()

        self.min_area = tk.Scale(self.frame, from_=1, to=6000, orient=tk.HORIZONTAL, label="min_area",
                                 command=self.update_image, length=600)
        self.min_area.set(args.min_area)
        self.min_area.pack()

        self.max_area = tk.Scale(self.frame, from_=1, to=6000, orient=tk.HORIZONTAL, label="max_area",
                                 command=self.update_image, length=600)
        self.max_area.set(args.max_area)
        self.max_area.pack()

        self.min_ratio = tk.Scale(self.frame, from_=0.0, to=1.5, resolution=0.05, orient=tk.HORIZONTAL, label="min_ratio",
                                  command=self.update_image, length=600)
        self.min_ratio.set(args.min_ratio)
        self.min_ratio.pack()

        self.max_ratio = tk.Scale(self.frame, from_=0.0, to=1.5, resolution=0.05, orient=tk.HORIZONTAL, label="max_ratio",
                                  command=self.update_image, length=600)
        self.max_ratio.set(args.max_ratio)
        self.max_ratio.pack()

        self.min_circularity = tk.Scale(self.frame, from_=0.0, to=1, resolution=0.05, orient=tk.HORIZONTAL, label="min_circularity",
                                        command=self.update_image, length=600)
        self.min_circularity.set(args.min_circularity)
        self.min_circularity.pack()

        self.max_circularity = tk.Scale(self.frame, from_=0.0, to=1, resolution=0.05, orient=tk.HORIZONTAL, label="max_circularity",
                                        command=self.update_image, length=600)
        self.max_circularity.set(args.max_circularity)
        self.max_circularity.pack()

        self.min_convexity = tk.Scale(self.frame, from_=0.0, to=1, resolution=0.05, orient=tk.HORIZONTAL, label="min_convexity",
                                      command=self.update_image, length=600)
        self.min_convexity.set(args.min_convexity)
        self.min_convexity.pack()

        self.max_convexity = tk.Scale(self.frame, from_=0.0, to=1, resolution=0.05, orient=tk.HORIZONTAL, label="max_convexity",
                                      command=self.update_image, length=600)
        self.max_convexity.set(args.max_convexity)
        self.max_convexity.pack()
        #
        # self.min_inertia_ratio = tk.Scale(self.frame, from_=0.0, to=1, resolution=0.05, orient=tk.HORIZONTAL, label="min_inertia_ratio", command=self.update_image, length=600)
        # self.min_inertia_ratio.set(args.min_inertia_ratio)
        # self.min_inertia_ratio.pack()
        #
        # self.max_inertia_ratio = tk.Scale(self.frame, from_=0.0, to=1, resolution=0.05, orient=tk.HORIZONTAL, label="max_inertia_ratio", command=self.update_image, length=600)
        # self.max_inertia_ratio.set(args.max_inertia_ratio)
        # self.max_inertia_ratio.pack()
        #
        self.min_color = tk.Scale(self.frame, from_=0, to=255, orient=tk.HORIZONTAL, label="min_color",
                                  command=self.update_image, length=600)
        self.min_color.set(args.min_color)
        self.min_color.pack()
        #
        self.max_color = tk.Scale(self.frame, from_=0, to=255, orient=tk.HORIZONTAL, label="max_color",
                                  command=self.update_image, length=600)
        self.max_color.set(args.max_color)
        self.max_color.pack()
        #
        self.min_dist_between_blobs = tk.Scale(self.frame, from_=1, to=100, orient=tk.HORIZONTAL,
                                               label="min_dist_between_blobs", command=self.update_image, length=600)
        self.min_dist_between_blobs.set(args.min_dist_between_blobs)
        self.min_dist_between_blobs.pack()

        self.min_pixel_value = tk.Scale(self.frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                        label="min_pixel_value", command=self.update_image, length=600)
        self.min_pixel_value.set(args.min_pixel_value)
        self.min_pixel_value.pack()

        self.max_pixel_value = tk.Scale(self.frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                        label="max_pixel_value", command=self.update_image, length=600)
        self.max_pixel_value.set(args.max_pixel_value)
        self.max_pixel_value.pack()

        # create a button to go to the next image
        self.next_button = tk.Button(self.root, text="Next", command=self.next_image)
        self.next_button.pack()

        # create a button to go to the previous image
        self.prev_button = tk.Button(self.root, text="Previous", command=self.prev_image)
        self.prev_button.pack()

        # create a button to quit
        self.quit_button = tk.Button(self.root, text="Quit", command=self.root.destroy)
        self.quit_button.pack()

        # create a button to save the parameters
        self.save_button = tk.Button(self.root, text="Save", command=self.save_parameters)
        self.save_button.pack()

        # create a button to load the parameters
        self.load_button = tk.Button(self.root, text="Load", command=self.load_parameters)
        self.load_button.pack()

        self.root.bind('<Left>', self.prev_image)
        self.root.bind('<Right>', self.next_image)
        self.root.bind('<Escape>', self.quit)

    def set_PIL_image(self, index):
        # image from numpy array
        self.image = cv2.imread(self.filenames[index], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(img, self.contours[index], -1, (0, 0, 255), 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(img_rgb)

    def next_image(self):
        self.index += 1
        if self.index >= len(self.filenames):
            self.index = len(self.filenames) - 1
        # image from numpy array after contour detection
        self.set_PIL_image(self.index)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def prev_image(self):
        self.index -= 1
        self.index = max(self.index, 0)
        # image from numpy array after contour detection
        self.set_PIL_image(self.index)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def update_image(self, event=None):
        # update each parameter
        self.args.blur = self.blur.get()
        self.args.canny = self.canny.get()
        self.args.dilate = self.dilate.get()
        self.args.erode = self.erode.get()
        self.args.min_area = self.min_area.get()
        self.args.max_area = self.max_area.get()
        self.args.min_ratio = self.min_ratio.get()
        self.args.max_ratio = self.max_ratio.get()
        self.args.min_circularity = self.min_circularity.get()
        self.args.max_circularity = self.max_circularity.get()
        # self.args.min_inertia_ratio = self.min_inertia_ratio.get()
        # self.args.max_inertia_ratio = self.max_inertia_ratio.get()
        self.args.min_pixel_value = self.min_pixel_value.get()
        self.args.max_pixel_value = self.max_pixel_value.get()
        self.args.min_convexity = self.min_convexity.get()
        self.args.max_convexity = self.max_convexity.get()
        self.args.min_color = self.min_color.get()
        self.args.max_color = self.max_color.get()

        # self.args.min_threshold = self.min_threshold.get()
        # self.args.max_threshold = self.max_threshold.get()
        # self.args.threshold_step = self.threshold_step.get()

        for k, v in self.args.__dict__.items():
            print(f"{k} : {v}")
        print()

        # filter by area
        # self.args.filter_by_area = self.filter_by_area.get()

        # # filter by ratio
        # self.args.filter_by_ratio = self.filter_by_ratio.get()
        # self.args.min_ratio = self.min_ratio.get()
        # self.args.max_ratio = self.max_ratio.get()
        #
        # # filter by color
        # self.args.filter_by_color = self.filter_by_color.get()
        # self.args.blob_color = self.blob_color.get()
        # self.args.min_color = self.min_color.get()
        #
        # # filter by circularity
        # self.args.filter_by_circularity = self.filter_by_circularity.get()
        # self.args.min_circularity = self.min_circularity.get()
        # self.args.max_circularity = self.max_circularity.get()
        #
        # # filter by convexity
        # self.args.filter_by_convexity = self.filter_by_convexity.get()
        # self.args.min_convexity = self.min_convexity.get()
        # self.args.max_convexity = self.max_convexity.get()
        #
        # # filter by inertia
        # self.args.filter_by_inertia = self.filter_by_inertia.get()
        # self.args.min_inertia_ratio = self.min_inertia_ratio.get()
        # self.args.max_inertia_ratio = self.max_inertia_ratio.get()

        # filter by distance between blobs
        # self.args.min_dist_between_blobs = self.min_dist_between_blobs.get()

        self.contours = get_contours_from_images(load_images(self.path), self.args)

        # update the image

        # self.contours = [filter_by_area(contours, self.args) for contours in self.contours]
        # self.contours = [filter_by_ratio(contours, self.args) for contours in self.contours]
        # self.contours = [filter_by_color(contours, self.args) for contours in self.contours]
        # self.contours = [filter_by_min_dist_between_blobs(contours, self.args) for contours in self.contours]
        # self.contours = [filter_by_circularity(contours, self.args) for contours in self.contours]
        # self.contours = [filter_by_convexity(contours, self.args) for contours in self.contours]
        # self.contours = [filter_by_inertia(contours, self.args) for contours in self.contours]

        # image from numpy array
        self.image = cv2.imread(self.filenames[self.index], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(img, self.contours[self.index], -1, (0, 0, 255), 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(img_rgb)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def save_parameters(self):
        # get the values from the sliders
        params = self.sliders.get()

        # save the values to a file
        with open('params.json', 'w') as f:
            json.dump(params, f)

    def load_parameters(self):
        # load the values from a file
        with open('params.json', 'r') as f:
            params = json.load(f)

        # set the values of the sliders
        self.sliders.set(params)

    def quit(self, event):
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    # img = load_image("example.png")
    # contours = get_contours(img)
    # plot_red_contours(contours, img)

    args = parse_args()

    path = "../upwork/bio-band-detection/inputs/"
    images = load_images(path)
    images_ = load_image_filenames(path)

    contours = get_contours_from_images(images, args)

    # # plot the first image with its contours
    # plot_red_contours(contours[0], images[0])

    gui = myGUI(path=path, filenames=images_, args=args)
    gui.run()


if __name__ == "__main__":
    main()

# todo: update with the click of a button and a key like space.
# when saving, it will store the points, the params and the area and pixels value.
