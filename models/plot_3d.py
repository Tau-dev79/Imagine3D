import trimesh
import pyrender

class PLOT3D:
    def plot(self, path):
        mesh = trimesh.load(path)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        render_mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene()
        scene.add(render_mesh)
        camera = pyrender.PerspectiveCamera(yfov=0.5)
        cam_pose = [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 6.0],
                    [0, 0, 0, 1]]
        scene.add(camera, pose=cam_pose)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=cam_pose)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)