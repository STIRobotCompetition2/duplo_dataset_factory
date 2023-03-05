import trimesh
import numpy as np
import os
from util.generate_construction import generate_construction
import cv2 as cv
import io
from PIL import Image
import matplotlib.pyplot as plt
from trimesh.transformations import transform_points
from scipy.spatial.transform import Rotation as R
import time
import pyrender

def generate_sample(id:int, dir:str="generated", max_height:int=5, noise_intensity:float=0.0, timing:bool=False, empty_prob:float=0.001): 
    if np.random.rand() < empty_prob:
        print(noise_intensity)
        empty_image = np.ones((640,640,3), dtype=np.uint8) 
        empty_image[:,:] = np.random.normal(255 * 0.2,255 * 0.05,3).astype(np.uint8)
        if noise_intensity > 0.0:
            row,col,ch= empty_image.shape
            gauss = np.random.normal(0.0,(noise_intensity)**0.5,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            empty_image = empty_image + gauss.astype(np.uint8)
        cv.imwrite("{}/images/im{}.jpg".format(dir, id), empty_image)
        open("{}/labels/im{}.txt".format(dir, id), "w+").close()

    # Generate Construction
    if timing: start_time=time.time()
    construction = generate_construction(np.random.randint(1,max_height+1),0,0,0)
    if timing: print("[Construction-Generation]\t{} s".format(time.time()-start_time))
    
    if timing: start_time=time.time()
    # Generate Scene
    scene = trimesh.Scene()
    # Insert Construction in scene
    for layer in construction:
        for brick in layer:
            scene.add_geometry(brick[0])
    if timing: print("[Construction2Scene]\t{} s".format(time.time()-start_time))
    # Generate Random Spatial Transformation for DUPLO construction
    random_rot = R.from_euler("xyz", np.random.normal(0.0, 3*np.pi/2, size=(3))).as_matrix()
    random_trans = np.diag([2,2,8]) @ np.random.rand(3) + np.array([-1,-1,-10])
    construction_pose = np.eye(4)
    construction_pose[:3,:3] = random_rot
    construction_pose[:3,3] = random_trans
    # Apply this transform
    c = scene.camera_transform
    scene.apply_transform(construction_pose)
    scene.camera_transform = c
    potential_border_points = np.array(scene.convex_hull.vertices).transpose()

    # Insert Ground-Plane such that lowest point of construction touches it
    lowest_z = np.min(potential_border_points[2,:])
    bottom_plane_tf = np.eye(4)
    bottom_plane_tf[:3,3] = np.array([0,0,lowest_z])
    bottom_plane = trimesh.primitives.Box((100,100, 0.01,), bottom_plane_tf)
    uv = np.random.rand(bottom_plane.vertices.shape[0], 2)
    # Ground plane has random greyish shape for now
    bottom_plane.visual = trimesh.visual.TextureVisuals(uv=uv,material=trimesh.visual.material.SimpleMaterial(diffuse=np.abs(np.random.normal(0.2,0.05,3)), glossiness=0, specular=np.abs(np.random.normal(0.2,0.05,3))))
    scene.add_geometry(bottom_plane)
    
    # Construct Pyrender-Scene from trimesh
    # TODO: Explain magic numbers (or compute them)
    if timing: start_time=time.time()
    pyrender_scene = pyrender.Scene.from_trimesh_scene(scene)
    cam = pyrender.IntrinsicsCamera(772.5483399, 772.5483399, 320, 320, scene.camera.z_near, scene.camera.z_far)
    cam_node = pyrender_scene.add(cam, pose=scene.camera_transform)
    dl = pyrender.DirectionalLight(color=np.clip(np.random.normal(0.9,0.05,3),0,1.0), intensity=np.random.normal(7.0, 5.0),)
    pyrender_scene.add(dl)
    renderer = pyrender.offscreen.OffscreenRenderer(640, 640)
    # Rendering Time!
    rendered_pyrender, _ = renderer.render(pyrender_scene, pyrender.RenderFlags.SHADOWS_ALL )
    if timing: print("[Rendering]\t{} s".format(time.time()-start_time))

    rendered_pyrender = rendered_pyrender[::-1]
    if timing: start_time=time.time()

    # Extract Bounding Box from 'most excentric' points of convex hull
    potential_border_points = np.concatenate((potential_border_points, np.ones((1,potential_border_points.shape[1]))), axis=0)
    K = np.array([
        [772.5483399, 0, 320],
        [0, 772.5483399, 320],
        [0, 0, 1],

    ])
    # Transform points into imaging plane
    potential_border_points =  K @ (np.linalg.inv(scene.camera_transform) @ potential_border_points)[:-1,:]
    potential_border_points = potential_border_points[:2, :] / potential_border_points[2:, :]
    potential_border_points[0,:] = 640 - potential_border_points[0,:]
    potential_border_points[1,:] = 640 - potential_border_points[1,:]
    potential_border_points = np.clip(potential_border_points, 0, 639)
    # Find extrema = Corners of BB-rectangle
    upper_border = np.max(potential_border_points, axis=1)
    lower_border = np.min(potential_border_points, axis=1)

    if noise_intensity > 0.0:
        row,col,ch= rendered_pyrender.shape
        gauss = np.random.normal(0.0,noise_intensity**0.5,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        rendered_pyrender = rendered_pyrender + gauss
    if timing: print("[Projection]\t{} s".format(time.time()-start_time))

    # Write results to disk
    if timing: start_time=time.time()
    cv.imwrite("{}/images/im{}.jpg".format(dir, id), rendered_pyrender)
    cv.rectangle(rendered_pyrender, lower_border.astype(int).tolist(), upper_border.astype(int).tolist(), color=[255,0,255], thickness=5)
    cv.imwrite("{}/images/im{}_withBB.jpg".format(dir, id), rendered_pyrender)
    with open("{}/labels/im{}.txt".format(dir, id), "w+") as label_file:
        upper_border /= 640.0
        lower_border /= 640.0
        entry = "0 {} {} {} {}".format(*(upper_border + lower_border)/2, *(upper_border - lower_border))
        label_file.write(entry)
    if timing: print("[Write2Disk]\t{} s".format(time.time()-start_time))
    

if __name__=="__main__":
    import sys
    try:
        id = int(sys.argv[1])
    except:
        id = 0
    try:
        max_height = int(sys.argv[2])
    except:
        max_height = 0
    try:
        noise_intensity = int(sys.argv[3])
    except:
        noise_intensity = 0
    print("Generate sample with id {} and max_height {} and noise_intensity {}".format(id, max_height, noise_intensity))
    generate_sample(id=id, max_height=max_height, noise_intensity=noise_intensity)     
    


