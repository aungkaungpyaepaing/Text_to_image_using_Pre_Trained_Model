Text-to-image generators have gained significant popularity recently, with many enthusiasts eager to experiment with this fascinating technology. In this blog post, I will guide you through the process of creating your own text-to-image generator using a pre-trained model that is freely available.
Before running the code, change the runtime type to GPU.
<h1>Install Requires Libraries</h1>
!pip install diffusers --upgrade: Upgrades the diffusers library to the latest version.
!pip install invisible_watermark transformers accelerate safetensors: Install additional libraries required for the model.
pip install torch torchvision torchaudio: Installs PyTorch and related libraries.

<h1>Import the model</h1>
import torch: Imports the PyTorch library.
from diffusers import DiffusionPipeline: Imports the DiffusionPipeline class from the diffusers library.

<h1>Load the model</h1>
DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0"): Loads the Stable Diffusion model from the specified repository.
torch_dtype=torch.float16: Uses 16-bit floating-point precision for the model.
use_safetensors=True: Uses the safetensors format for loading the model.
variant="fp16": Specifies the variant of the model to use.

<h1>Move the model to GPU</h1>
pipe.to("cuda"): Moves the model to the GPU for faster computation.

<h1>Generating image</h1>
prompt = "a man riding a horse on the moon": Defines the text prompt that describes the image to be generated.
pipe(prompt=prompt): Generates images based on the provided prompt.
.images[0]: Extracts the first generated image from the result.

![image](https://github.com/aungkaungpyaepaing/Text_to_image_using_Pre_Trained_Model/assets/88584127/7838d50d-954d-4e73-90d1-393a3048613c)

Kaggle link
https://medium.com/@kaungpyae.kp27/text-to-image-generator-using-a-pre-trained-model-a7ae0b6de475
