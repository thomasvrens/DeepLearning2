import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors

# data paths
PATH_RPLAN = r'C:\Users\caspervanengel\OneDrive\Documents\PHD\1_data\rplan\0-full'
PATH_LIFULL = r'C:\Users\caspervanengel\OneDrive\Documents\PHD\1_data\lifull'

# frequently used colors
delft_color = "#00A6D6"  # TU Delft light blue color
google_colors = ["#4285F4",  # blue
                 "#DB4437",  # red
                 "#F4B400",  # yellow
                 "#0F9D58"  # green
                 ]

# coloring schemes
# COLORS_RPLAN = ['#e6550d',  # living room
#                 '#1f77b4',  # master room
#                 '#fd8d3c',  # kitchen
#                 '#6b6ecf',  # bathroom
#                 '#fdae6b',  # dining room
#                 '#d3d3d3',  # child room
#                 '#d3d3d3',  # study room
#                 '#1f77b4',  # second room
#                 '#1f77b4',  # guest room
#                 '#2ca02c',  # balcony
#                 '#fdd0a2',  # entrance
#                 '#5254a3',  # storage
#                 '#5254a3',  # walk-in
#                 '#ffffff',  # external area
#                 '#000000',  # exterior wall
#                 '#ffa500',  # front door
#                 '#000000',  # interior wall
#                 '#ff0000']  # interior door

COLORS_RPLAN = ['#e6550d',  # living room
                '#1f77b4',  # master room
                '#fd8d3c',  # kitchen
                '#6b6ecf',  # bathroom
                '#fdae6b',  # dining room
                '#d3d3d3',  # child room
                '#d3d3d3',  # study room
                '#1f77b4',  # second room
                '#1f77b4',  # guest room
                '#2ca02c',  # balcony
                '#fdd0a2',  # entrance
                '#5254a3',  # storage
                '#5254a3',  # walk-in
                '#000000',  # external area
                '#ffffff',  # exterior wall
                '#ffa500',  # front door
                '#ffffff',  # interior wall
                '#ff0000']  # interior door

COLORS_LIFULL = ['#e6550d',  # living room
                 '#fd8d3c',  # kitchen
                 '#1f77b4',  # bedroom
                 '#6b6ecf',  # bathroom
                 '#808080',  # missing (todo: better color)
                 '#5254a3',  # closet
                 '#2ca02c',  # balcony
                 '#fdd0a2',  # corridor
                 '#fdae6b',  # dining room
                 '#d3d3d3']  # laundry room (todo: better color)

COLORS_MSD = ['#1f77b4',  # bedroom
              '#e6550d',  # living room
              '#fd8d3c',  # kitchen
              '#fdae6b',  # dining
              '#fdd0a2',  # corridor
              '#72246c',  # stairs
              '#5254a3',  # storeroom
              '#6b6ecf',  # bathroom
              '#2ca02c',  # balcony
              '#000000',  # structure
              '#ffc000',  # door
              '#98df8a',  # entrance door
              '#d62728']  # window

# color maps
CMAP_RPLAN = get_cmap(mcolors.ListedColormap(COLORS_RPLAN))
CMAP_LIFULL = get_cmap(mcolors.ListedColormap(COLORS_LIFULL))
CMAP_MSD = get_cmap(mcolors.ListedColormap(COLORS_MSD))

# categories (rooms and doors)
CAT_RPLAN = ["living room",  # rplan: 45k version
             "master room",
             "kitchen",
             "bathroom",
             "dining room",
             "child room",
             "study room",
             "second room",
             "guest room",
             "balcony",
             "entrance",
             "storage",
             "walk-in",
             "external area",
             "exterior wall",
             "front door",
             "interior wall",
             "interior door"]

CAT_LIFULL = ["living_room",  # lifull: 145k HouseGAN version
              "kitchen",
              "bedroom",
              "bathroom",
              "missing",
              "closet",
              "balcony",
              "corridor",
              "dining_room",
              "laundry_room"]

CAT_MSD = ['Bedroom',  # modified swiss dwellings: 5.4k V2 version
           'Livingroom',
           'Kitchen',
           'Dining',
           'Corridor',
           'Stairs',
           'Storeroom',
           'Bathroom',
           'Balcony',
           'Structure',
           'Door',
           'Entrance Door',
           'Window']


# classes
CLASSES_RPLAN = np.arange(0, len(CAT_RPLAN))
CLASSES_LIFULL = np.arange(0, len(CAT_LIFULL))
CLASSES_MSD = np.arange(0, len(CAT_LIFULL))


# class-category mapping
MAPPING_RPLAN = {n: cat for n, cat in enumerate(CAT_RPLAN)}
MAPPING_LIFULL = {n: cat for n, cat in enumerate(CAT_LIFULL)}
MAPPING_MSD = {n: cat for n, cat in enumerate(CAT_MSD)}