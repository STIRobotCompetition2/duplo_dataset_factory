import trimesh
import numpy as np
import os
from generate_construction import generate_construction
import cv2 as cv
import io
from PIL import Image
import matplotlib.pyplot as plt
from trimesh.transformations import transform_points
from scipy.spatial.transform import Rotation as R
import pyrender


roll, pitch, yaw = np.random.normal(0.0, 3*np.pi/2, size=(3))

random_rot = R.from_euler("xyz", np.random.normal(0.0, 3*np.pi/2, size=(3))).as_matrix()
random_trans = np.diag([2,2,8]) @ np.random.rand(3) + np.array([-1,-1,-10])

camera_pose = np.eye(4)
camera_pose[:3,:3] = random_rot
camera_pose[:3,3] = random_trans


construction = generate_construction(2,0,0,0)
scene = trimesh.Scene()
for layer in construction:
    for brick in layer:
        scene.add_geometry(brick[0])
if not os.path.isdir("generated"): os.mkdir("generated")
c = scene.camera_transform
scene.apply_transform(camera_pose)
scene.camera_transform = c
potential_border_points = np.array(scene.convex_hull.vertices).transpose()

lowest_z = np.min(potential_border_points[2,:])
bottom_plane_tf = np.eye(4)
bottom_plane_tf[:3,3] = np.array([0,0,lowest_z])
bottom_plane = trimesh.primitives.Box((100,100, 0.01,), bottom_plane_tf)
uv = np.random.rand(bottom_plane.vertices.shape[0], 2)
bottom_plane.visual = trimesh.visual.TextureVisuals(uv=uv,material=trimesh.visual.material.SimpleMaterial(diffuse=[0.2,0.2,0.2], glossiness=0, specular=[0.2,0.2,0.2]))
scene.add_geometry(bottom_plane)

# image_bin = scene.save_image((640,640), background=[200,200,0,255])
# image_cv = np.array(Image.open(io.BytesIO(image_bin)))

pyrender_scene = pyrender.Scene.from_trimesh_scene(scene)
cam = pyrender.IntrinsicsCamera(772.5483399, 772.5483399, 320, 320, scene.camera.z_near, scene.camera.z_far)
cam_node = pyrender_scene.add(cam, pose=scene.camera_transform)
# pyrender.Viewer(pyrender_scene, viewport_size=scene.camera.resolution,
#                 use_raymond_lighting=True)
dl = pyrender.DirectionalLight(color=[1.0, 1.0, 0.8], intensity=5.0,)

pyrender_scene.add(dl)
renderer = pyrender.offscreen.OffscreenRenderer(640,
                                                640)
rendered_pyrender, _ = renderer.render(pyrender_scene, pyrender.RenderFlags.SHADOWS_ALL )
rendered_pyrender = rendered_pyrender[::-1]



potential_border_points = np.concatenate((potential_border_points, np.ones((1,potential_border_points.shape[1]))), axis=0)
# Transform into image coordinate system
transformed = transform_points(scene.convex_hull.vertices, np.linalg.inv(scene.camera_transform))
projected = transformed @ scene.camera.K.T
K = np.array([
    [772.5483399, 0, 320],
    [0, 772.5483399, 320],
    [0, 0, 1],

])
K[0,0] = K[1,1]
potential_border_points =  K @ (np.linalg.inv(scene.camera_transform) @ potential_border_points)[:-1,:]

potential_border_points = potential_border_points[:2, :] / potential_border_points[2:, :]
potential_border_points[0,:] = 640 - potential_border_points[0,:]
potential_border_points[1,:] = 640 - potential_border_points[1,:]

potential_border_points = np.clip(potential_border_points, 0, 639) # Constrain
upper_border = np.max(potential_border_points, axis=1)
lower_border = np.min(potential_border_points, axis=1)

image_cv = cv.rectangle(rendered_pyrender, lower_border.astype(int).tolist(), upper_border.astype(int).tolist(), color=[255,0,255], thickness=5)
plt.imshow(image_cv)
plt.show()




image_cv = np.array(rendered_pyrender)
print(scene.camera.K)
# cv.imshow("TEST", image_cv)
# cv.waitKey()
cv.imwrite("generated/test.png", image_cv)
    
    



