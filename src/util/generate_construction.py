import trimesh
import numpy as np
import os
import copy
import logging 
import csv
from textwrap import wrap

DUPLO_WIDTH = 3.18 / 2
DUPLO_HEIGHT = 1.92

COLOR_PALETTE = [
    [255, 0, 0, 255],
    [0, 255, 0, 255],
    [0, 0, 255, 255]
]

BRICK_TYPES = [
    [trimesh.primitives.Box([3,3,3], np.array([[1,0,0,0.75],[0,1,0,0.75],[0,0,1,1.5],[0,0,0,1]])), [2,2], [0,0,0]],
    [trimesh.primitives.Box([6,3,3], np.array([[1,0,0,2.25],[0,1,0,0.75],[0,0,1,1.5],[0,0,0,1]])), [4,2], [0,0,0]],
]

def prepare_database(brick_dir:str="data/bricks", color_file:str="data/colors/colors.csv"):
    brick_files = [f for f in os.listdir(brick_dir) if os.path.isfile(os.path.join(brick_dir, f)) ]
    brick_geometries = []
    scale = -1
    for brick_file in brick_files:
        new_brick = [
            trimesh.load(os.path.join(brick_dir, brick_file)),
            [int(dim) for dim in brick_file[:-4].split("x", 1)],
            [0,0,0]
        ]

        test_scale = np.divide(
            np.array(np.max(new_brick[0].bounding_box.vertices, axis=0)[0:2]) - np.array(np.min(new_brick[0].bounding_box.vertices, axis=0)[0:2]),
            np.array(new_brick[1]))
        assert int(test_scale[0]) == int(test_scale[1])
        if scale<0: scale = int(test_scale[0])
        else: assert int(test_scale[0]) == scale
        brick_geometries.append(new_brick)
        new_brick[0] = new_brick[0].apply_scale(DUPLO_WIDTH / scale)
    color_palette = []
    with open(color_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            try: color_palette.append([int(hex_as_str, 16) for hex_as_str in wrap(row[2],2)])
            except: pass
    logging.warn("Color loading not yet supported")
    return brick_geometries, color_palette



def rotation_matrix_2d(theta:float): return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def transform(base_mesh: list, new_mesh: list, connection:list, layer:int):
    original_tf = base_mesh[2]
    planar_tf = np.array([connection[0], connection[1]]) * DUPLO_WIDTH
    planar_tf = rotation_matrix_2d(original_tf[2]) @ planar_tf + np.array(original_tf[0:2])
    
    new_mesh[2] = [*planar_tf.tolist(), connection[2] * np.pi / 2]
    
    new_mesh[0].apply_transform(np.array([
        [np.cos(new_mesh[2][2]), -np.sin(new_mesh[2][2]), 0, new_mesh[2][0]],
        [np.sin(new_mesh[2][2]), np.cos(new_mesh[2][2]), 0, new_mesh[2][1]],
        [0, 0, 1, layer * DUPLO_HEIGHT],
        [0, 0, 0, 1],
    ]))
    return new_mesh
    

def generate_construction(brick_shapes:list, color_palette:list, max_height:int)->trimesh.Trimesh:
    layers = [[copy.deepcopy(brick_shapes[0])]]
    uv = np.random.rand(layers[0][0][0].vertices.shape[0], 2)
    next_color = color_palette[np.random.randint(0,len(color_palette))]
    layers[0][0][0].visual = trimesh.visual.TextureVisuals(uv=uv,material=trimesh.visual.material.SimpleMaterial(diffuse=next_color, glossiness=1, ambient=next_color, specular=next_color))
    while len(layers) < max_height + 1:
        new_layer = []
        for brick in layers[-1]:
            while np.random.rand() > np.exp(0.2 * len(new_layer)) - 1:
                next_brick = copy.deepcopy(brick_shapes[np.random.randint(0,len(BRICK_TYPES))])
    
                next_color = color_palette[np.random.randint(0,len(color_palette))]
                # next_brick = copy.deepcopy(BRICK_TYPES[0])
                uv = np.random.rand(next_brick[0].vertices.shape[0], 2)
                next_brick[0].visual = trimesh.visual.TextureVisuals(uv=uv,material=trimesh.visual.material.SimpleMaterial(diffuse=next_color, glossiness=1, ambient=None, specular=next_color))
                next_connection = [np.random.randint(0, brick[1][0]), np.random.randint(0, brick[1][1]), np.random.randint(0,4)]
                transform(brick, next_brick, next_connection, len(layers))
                # next_brick[0] = trimesh.Trimesh(next_brick[0].vertices, next_brick[0].faces, visual=trimesh.visual.TextureVisuals(face_materials=trimesh.visual.material.SimpleMaterial(diffuse=next_color, glossiness=1.0, ambient=next_color)))
                new_layer.append(copy.deepcopy(next_brick))
        layers.append(new_layer)
    return layers

if __name__ == "__main__":
    data_set,color_palette = prepare_database()
    construction = generate_construction(data_set, color_palette, 5)
    scene = trimesh.Scene()
    scene.add_geometry(data_set[-1][0])


    for layer in construction:
        for brick in layer:
            scene.add_geometry(brick[0])
    scene.show()


