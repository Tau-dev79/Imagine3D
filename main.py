import warnings
warnings.filterwarnings("ignore")
import torch
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from models.image_to_3d import IMG3D 
from models.prompt_to_3d import PROMPT3D 
from models.plot_3d import PLOT3D
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
image_model = load_model('image300M', device=device)
text_model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

if __name__ == "__main__":
    image_to_3d = IMG3D(image_model, diffusion, xm)
    prompt_to_3d = PROMPT3D(text_model, diffusion, xm)
    plot_obj = PLOT3D()
    try:
        while True:
            print("""
Choose your option:

1. Generate from image.
2. Generate from prompt.
3. Plot 3d Object
            """)
            option = int(input(""))
            match option:
                case 1:
                    files = os.listdir("example")
                    for i, file in enumerate(files):
                        print(f"{i+1}. {file}")
                    file_id = int(input("Enter the file number you want to convert: "))
                    if file_id > len(files):
                        print("Invalid id selected")
                        break
                    print("Processing ...")
                    obj_path = image_to_3d.convert_3d(f"example/{files[file_id-1]}")
                    show_obj = int(input("Press 1 to plot the created object or 0 to exit: "))
                    if show_obj != 1:
                        print("Exiting!")
                        break
                    plot_obj.plot(obj_path)
                    continue
                case 2:
                    prompt = input("Enter your prompt: ")
                    print("Processing ...")
                    obj_path = prompt_to_3d.convert_3d(prompt)
                    show_obj = int(input("Press 1 to plot the created object or 0 to exit: "))
                    if show_obj != 1:
                        print("Exiting!")
                        break
                    plot_obj.plot(obj_path)
                    continue
                case 3:
                    files = os.listdir("outputs")
                    for i, file in enumerate(files):
                        print(f"{i+1}. {file}")
                    select = int(input("Enter the number of the file you want to plot: "))
                    if select > len(files):
                        print("Exiting!")
                        break
                    plot_obj.plot(f"outputs/{files[select-1]}")
                    continue
                case _:
                    print("Exiting!")
                    break


    except:
        print("Something went wrong!")