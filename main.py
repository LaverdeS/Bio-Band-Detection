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


def get_contours(img, args):
    # get contours
    img = cv2.medianBlur(img, args.blur)
    img = cv2.Canny(img, args.canny, args.canny * 2)
    img = cv2.dilate(img, None, iterations=args.dilate)
    img = cv2.erode(img, None, iterations=args.erode)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # filter contours
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if args.filter_by_area and (area < args.min_area or area > args.max_area):
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        ratio = 4 * np.pi * area / perimeter ** 2
        if args.filter_by_ratio and (ratio < args.min_ratio or ratio > args.max_ratio):
            continue
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        convexity = float(area) / hull_area
        if args.filter_by_convexity and (convexity < args.min_convexity or convexity > args.max_convexity):
            continue
        filtered_contours.append(contour)

    return filtered_contours


def get_contours_from_images(images, args):
    contours = []
    for img in images:
        contours.append(get_contours(img, args))
    return contours


def plot_red_contours(contours, img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blur", type=int, default=5)
    parser.add_argument("--canny", type=int, default=50)
    parser.add_argument("--dilate", type=int, default=1)
    parser.add_argument("--erode", type=int, default=1)
    parser.add_argument("--filter_by_area", action="store_true")
    parser.add_argument("--min_area", type=int, default=100)
    parser.add_argument("--max_area", type=int, default=100000)
    parser.add_argument("--filter_by_ratio", action="store_true")
    parser.add_argument("--min_ratio", type=float, default=0.5)
    parser.add_argument("--max_ratio", type=float, default=1.5)
    parser.add_argument("--filter_by_circularity", action="store_true")
    parser.add_argument("--min_circularity", type=float, default=0.5)
    parser.add_argument("--max_circularity", type=float, default=1.5)
    parser.add_argument("--filter_by_inertia", action="store_true")
    parser.add_argument("--min_inertia_ratio", type=float, default=0.5)
    parser.add_argument("--max_inertia_ratio", type=float, default=1.5)
    parser.add_argument("--filter_by_convexity", action="store_true")
    parser.add_argument("--min_convexity", type=float, default=0.5)
    parser.add_argument("--max_convexity", type=float, default=1.5)
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
        self.frame.pack()

        # create the sliders
        self.blur = tk.Scale(self.frame, from_=1, to=20, orient=tk.HORIZONTAL, label="blur",
                             command=self.update_image)
        self.blur.set(args.blur)
        # self.blur.bind("<ButtonRelease-1>", self.update_image())
        self.blur.pack()

        self.canny = tk.Scale(self.frame, from_=1, to=100, orient=tk.HORIZONTAL, label="canny")
        self.canny.set(args.canny)
        self.canny.pack()
        self.dilate = tk.Scale(self.frame, from_=1, to=10, orient=tk.HORIZONTAL, label="dilate")
        self.dilate.set(args.dilate)
        self.dilate.pack()
        self.erode = tk.Scale(self.frame, from_=1, to=10, orient=tk.HORIZONTAL, label="erode")
        self.erode.set(args.erode)
        self.erode.pack()
        self.filter_by_area = tk.IntVar()
        self.filter_by_area.set(args.filter_by_area)
        self.filter_by_area_check = tk.Checkbutton(self.frame, text="filter_by_area", variable=self.filter_by_area)
        self.filter_by_area_check.pack()
        self.min_area = tk.Scale(self.frame, from_=1, to=100000, orient=tk.HORIZONTAL, label="min_area")
        self.min_area.set(args.min_area)
        self.min_area.pack()
        self.max_area = tk.Scale(self.frame, from_=1, to=100000, orient=tk.HORIZONTAL, label="max_area")
        self.max_area.set(args.max_area)
        self.max_area.pack()
        self.filter_by_ratio = tk.IntVar()
        self.filter_by_ratio.set(args.filter_by_ratio)
        self.filter_by_ratio_check = tk.Checkbutton(self.frame, text="filter_by_ratio",
                                                    variable=self.filter_by_ratio)
        self.filter_by_ratio_check.pack()
        self.min_ratio = tk.Scale(self.frame, from_=0.1, to=10, orient=tk.HORIZONTAL, label="min_ratio")
        self.min_ratio.set(args.min_ratio)
        self.min_ratio.pack()
        self.max_ratio = tk.Scale(self.frame, from_=0.1, to=10, orient=tk.HORIZONTAL, label="max_ratio")
        self.max_ratio.set(args.max_ratio)
        self.max_ratio.pack()
        self.filter_by_circularity = tk.IntVar()
        self.filter_by_circularity.set(args.filter_by_circularity)
        self.filter_by_circularity_check = tk.Checkbutton(self.frame, text="filter_by_circularity",
                                                          variable=self.filter_by_circularity)
        self.filter_by_circularity_check.pack()
        self.min_circularity = tk.Scale(self.frame, from_=0.1, to=10, orient=tk.HORIZONTAL, label="min_circularity")
        self.min_circularity.set(args.min_circularity)
        self.min_circularity.pack()
        # ...

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

    def update_image(self, event):
        # update each parameter
        print("get_blur ", self.blur.get())
        self.args.blur = self.blur.get()

        self.contours = get_contours_from_images(load_images(self.path), self.args)
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
