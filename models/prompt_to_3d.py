
from shap_e.diffusion.sample import sample_latents
from shap_e.util.notebooks import decode_latent_mesh
import os

class PROMPT3D:
    def __init__(self, model, diffusion, xm, guidance_scale=15.0):
        self.model = model
        self.diffusion = diffusion
        self.xm = xm
        self.guidance_scale = guidance_scale
    
    def convert_3d(self, prompt):
        OUTPUT_PATH = f'outputs/mesh_{len(os.listdir("outputs"))}.obj'
        latents = sample_latents(
            batch_size=1,
            model=self.model,
            diffusion=self.diffusion,
            guidance_scale=self.guidance_scale,
            model_kwargs=dict(texts=[prompt]),
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