
from shap_e.diffusion.sample import sample_latents
from shap_e.util.notebooks import decode_latent_mesh
from shap_e.util.image_util import load_image
from rembg import remove
import os

class IMG3D:
    def __init__(self, model, diffusion, xm, guidance_scale=3.0):
        self.model = model
        self.diffusion = diffusion
        self.xm = xm
        self.guidance_scale = guidance_scale
    
    def remove_bg(self, input_path, output_path):
        with open(input_path, 'rb') as f:
            input_data = f.read()
        output_data = remove(input_data)
        with open(output_path, 'wb') as f:
            f.write(output_data)
        print(f"Background removed and saved to {output_path}")
        return output_path
    
    def convert_3d(self, path):
        OUTPUT_PATH = f'outputs/mesh_{len(os.listdir("outputs"))}.obj'
        cleaned_path = self.remove_bg(path, f'cleaned_image/clean_{len(os.listdir("outputs"))}.png')
        image = load_image(cleaned_path)
        latents = sample_latents(
            batch_size=1,
            model=self.model,
            diffusion=self.diffusion,
            guidance_scale=self.guidance_scale,
            model_kwargs=dict(images=[image]),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        for i, latent in enumerate(latents):
            t = decode_latent_mesh(self.xm, latent).tri_mesh()
            with open(OUTPUT_PATH, 'w') as f:
                t.write_obj(f)
        print("Object file has been created")
        return OUTPUT_PATH