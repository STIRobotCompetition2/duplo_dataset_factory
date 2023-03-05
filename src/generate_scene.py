import trimesh
import numpy as np
import os
from generate_construction import generate_construction
import cv2 as cv
import io
from PIL import Image
import matplotlib.pyplot as plt
from trimesh.transformations import transform_points


camera_pose = np.array([
    [0, -1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1]
])

construction = generate_construction(5,0,0,0)
scene = trimesh.Scene()

for layer in construction:
    for brick in layer:
        scene.add_geometry(brick[0])
if not os.path.isdir("generated"): os.mkdir("generated")
points = trimesh.PointCloud(scene.convex_hull.vertices)
scene.add_geometry(points)
scene.show()
image_bin = scene.save_image((1000,500))

potential_border_points = np.array(scene.convex_hull.vertices).transpose()
potential_border_points = np.concatenate((potential_border_points, np.ones((1,potential_border_points.shape[1]))), axis=0)
# Transform into image coordinate system
transformed = transform_points(scene.convex_hull.vertices, np.linalg.inv(scene.camera_transform))
projected = transformed @ scene.camera.K.T
K = scene.camera.K
K[0,0] = K[1,1]
potential_border_points =  K @ (np.linalg.inv(scene.camera_transform) @ potential_border_points)[:-1,:]
print(scene.camera_transform)

potential_border_points = potential_border_points[:2, :] / potential_border_points[2:, :]



image_cv = np.array(Image.open(io.BytesIO(image_bin)))
plt.imshow(image_cv)
plt.scatter(1000- potential_border_points[0,:], potential_border_points[1,:] - 1)
plt.show()




image_cv = np.array(Image.open(io.BytesIO(image_bin)))
print(scene.camera.focal)
# cv.imshow("TEST", image_cv)
# cv.waitKey()

with open("generated/test.png", "wb") as img_file:
    img_file.write(image_bin)
    



