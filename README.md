# ???????? (????? ???? ????):
# pip install diffusers transformers accelerate scipy safetensors gradio opencv-python bitsandbytes einops controlnet_aux

import torch
import gradio as gr
from diffusers import (
    StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline,
    ControlNetModel, StableDiffusionControlNetPipeline
)
from controlnet_aux import OpenposeDetector, CannyDetector
from PIL import Image
import numpy as np
import cv2
import random

MODEL_TXT2IMG = "stabilityai/stable-diffusion-2-1-base"
MODEL_TXT2IMG_XL = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_INPAINT = "runwayml/stable-diffusion-inpainting"
MODEL_CONTROLNET_CANNY = "lllyasviel/sd-controlnet-canny"
MODEL_CONTROLNET_OPENPOSE = "lllyasviel/sd-controlnet-openpose"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ???????? ??????
pipe_txt2img = StableDiffusionPipeline.from_pretrained(
    MODEL_TXT2IMG, torch_dtype=torch.float16
).to(device)
pipe_txt2img.enable_attention_slicing()
pipe_txt2img.enable_xformers_memory_efficient_attention()

pipe_txt2img_xl = StableDiffusionPipeline.from_pretrained(
    MODEL_TXT2IMG_XL, torch_dtype=torch.float16
).to(device)
pipe_txt2img_xl.enable_attention_slicing()
pipe_txt2img_xl.enable_xformers_memory_efficient_attention()

pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_TXT2IMG, torch_dtype=torch.float16
).to(device)
pipe_img2img.enable_attention_slicing()
pipe_img2img.enable_xformers_memory_efficient_attention()

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_INPAINT, torch_dtype=torch.float16
).to(device)
pipe_inpaint.enable_attention_slicing()
pipe_inpaint.enable_xformers_memory_efficient_attention()

controlnet_canny = ControlNetModel.from_pretrained(
    MODEL_CONTROLNET_CANNY, torch_dtype=torch.float16
).to(device)
controlnet_openpose = ControlNetModel.from_pretrained(
    MODEL_CONTROLNET_OPENPOSE, torch_dtype=torch.float16
).to(device)

pipe_controlnet_canny = StableDiffusionControlNetPipeline.from_pretrained(
    MODEL_TXT2IMG, controlnet=controlnet_canny, torch_dtype=torch.float16
).to(device)
pipe_controlnet_canny.enable_attention_slicing()
pipe_controlnet_canny.enable_xformers_memory_efficient_attention()

pipe_controlnet_openpose = StableDiffusionControlNetPipeline.from_pretrained(
    MODEL_TXT2IMG, controlnet=controlnet_openpose, torch_dtype=torch.float16
).to(device)
pipe_controlnet_openpose.enable_attention_slicing()
pipe_controlnet_openpose.enable_xformers_memory_efficient_attention()

canny = CannyDetector()
openpose = OpenposeDetector()

def generate_seed():
    return random.randint(0, 2**32 - 1)

def txt2img_fn(prompt, steps, width, height, model_choice, seed, brightness, contrast, batch_size):
    seed = int(seed) if seed is not None else generate_seed()
    torch.manual_seed(seed)
    images = []
    pipe = pipe_txt2img if model_choice == "SD 2.1" else pipe_txt2img_xl
    for _ in range(batch_size):
        img = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
        ).images[0]

        img = img.convert("RGB")
        img_np = np.array(img).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        if brightness != 0:
            img_np = cv2.convertScaleAbs(img_np, alpha=1, beta=brightness)
        if contrast != 1:
            img_np = cv2.convertScaleAbs(img_np, alpha=contrast, beta=0)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_out = Image.fromarray(img_np)
        images.append(img_out)
    return images if batch_size > 1 else images[0]

def lowres_preview_fn(prompt):
    image = pipe_txt2img(prompt=prompt, num_inference_steps=10, width=256, height=256).images[0]
    return image

def img2img_fn(prompt, input_image, strength, steps, seed):
    seed = int(seed) if seed is not None else generate_seed()
    torch.manual_seed(seed)
    image = pipe_img2img(
        prompt=prompt, image=input_image, strength=strength, num_inference_steps=steps
    ).images[0]
    return image

def inpaint_fn(prompt, input_image, mask, steps, seed):
    seed = int(seed) if seed is not None else generate_seed()
    torch.manual_seed(seed)
    image = pipe_inpaint(
        prompt=prompt, image=input_image, mask_image=mask, num_inference_steps=steps
    ).images[0]
    return image

def controlnet_fn(prompt, input_image, steps, use_canny, use_openpose, seed):
    seed = int(seed) if seed is not None else generate_seed()
    torch.manual_seed(seed)
    img = input_image.convert("RGB")

    detected_maps = []
    if use_canny:
        np_img = np.array(img)
        detected_maps.append(canny(np_img))
    if use_openpose:
        detected_maps.append(openpose(img))

    if len(detected_maps) == 0:
        return None

    if use_canny and not use_openpose:
        pipe = pipe_controlnet_canny
        input_map = detected_maps[0]
    elif use_openpose and not use_canny:
        pipe = pipe_controlnet_openpose
        input_map = detected_maps[0]
    else:
        # ??? ?? ?? ?????? ????? ??? canny ?? ???? ????? ?????? ???????
        pipe = pipe_controlnet_canny
        combined_map = np.mean(np.stack(detected_maps), axis=0).astype(np.uint8)
        input_map = Image.fromarray(combined_map)

    image = pipe(prompt=prompt, image=input_map, num_inference_steps=steps).images[0]
    return image

def generate_video(images, output_path="output.mp4", fps=10):
    w, h = images[0].size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for img in images:
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video.write(cv_img)
    video.release()
    return output_path

with gr.Blocks(title="Leonardo AI Ultimate GPU") as demo:
    gr.Markdown("# ?? Leonardo AI Ultimate Clone with Advanced Features")

    with gr.Tab("??? Text to Image"):
        prompt = gr.Textbox(label="Prompt")
        steps = gr.Slider(10, 100, value=30, step=5, label="Steps")
        width = gr.Slider(256, 1024, value=512, step=64, label="Width")
        height = gr.Slider(256, 1024, value=512, step=64, label="Height")
        model_choice = gr.Radio(["SD 2.1", "SD XL"], value="SD 2.1", label="Model Choice")
        seed = gr.Number(value=generate_seed(), label="Seed", precision=0)
        brightness = gr.Slider(-100, 100, value=0, label="Brightness")
        contrast = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Contrast")
        batch_size = gr.Slider(1, 5, value=1, step=1, label="Batch Size")
        preview_btn = gr.Button("Preview (Low-res)")
        output = gr.Gallery(label="Generated Images", show_label=True, elem_id="gallery_txt2img")

        preview_btn.click(lambda p: lowres_preview_fn(p), inputs=[prompt], outputs=[output])
        generate_btn = gr.Button("Generate")
        generate_btn.click(
            txt2img_fn,
            inputs=[prompt, steps, width, height, model_choice, seed, brightness, contrast, batch_size],
            outputs=[output],
        )

    with gr.Tab("??? Image to Image"):
        prompt2 = gr.Textbox(label="Prompt")
        input_img2 = gr.Image(type="pil", label="Input Image")
        strength = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Strength")
        steps2 = gr.Slider(10, 100, value=30, step=5, label="Steps")
        seed2 = gr.Number(value=generate_seed(), label="Seed", precision=0)
        output2 = gr.Image(label="Output Image")
        btn2 = gr.Button("Generate")

        btn2.click(
            img2img_fn,
            inputs=[prompt2, input_img2, strength, steps2, seed2],
            outputs=[output2],
        )

    with gr.Tab("?? Inpainting"):
        prompt3 = gr.Textbox(label="Prompt")
        input_img3 = gr.Image(type="pil", label="Input Image")
        mask3 = gr.Image(type="pil", label="Mask Image")
        steps3 = gr.Slider(10, 100, value=30, step=5, label="Steps")
        seed3 = gr.Number(value=generate_seed(), label="Seed", precision=0)
        output3 = gr.Image(label="Output Image")
        btn3 = gr.Button("Generate")

        btn3.click(
            inpaint_fn,
            inputs=[prompt3, input_img3, mask3, steps3, seed3],
            outputs=[output3],
        )

    with gr.Tab("??? ControlNet (Canny + OpenPose)"):
        prompt4 = gr.Textbox(label="Prompt")
        input_img4 = gr.Image(type="pil", label="Input Image")
        steps4 = gr.Slider(10, 100, value=30, step=5, label="Steps")
        seed4 = gr.Number(value=generate_seed(), label="Seed", precision=0)
        use_canny = gr.Checkbox(value=True, label="Use Canny")
        use_openpose = gr.Checkbox(value=False, label="Use OpenPose")
        output4 = gr.Image(label="Output Image")
        btn4 = gr.Button("Generate")

        btn4.click(
            controlnet_fn,
            inputs=[prompt4, input_img4, steps4, use_canny, use_openpose, seed4],
            outputs=[output4],
        )

    with gr.Tab("??? Generate Video"):
        prompt_vid = gr.Textbox(label="Prompt")
        frames_vid = gr.Slider(1, 10, value=5, step=1, label="Number of Frames")
        steps_vid = gr.Slider(10, 100, value=30, step=5, label="Steps per Frame")
        seed_vid = gr.Number(value=generate_seed(), label="Seed", precision=0)
        output_vid = gr.File(label="Output Video")
        generate_video_btn = gr.Button("Generate Video")

        def create_video_from_prompt(prompt, frames, steps, seed):
            seed = int(seed) if seed is not None else generate_seed()
            torch.manual_seed(seed)
            images = []
            for i in range(frames):
                img = pipe_txt2img(prompt=prompt, num_inference_steps=steps, width=512, height=512).images[0]
                images.append(img)
            video_path = generate_video(images, output_path="output.mp4", fps=10)
            return video_path

        generate_video_btn.click(
            create_video_from_prompt,
            inputs=[prompt_vid, frames_vid, steps_vid, seed_vid],
            outputs=[output_vid]
        )

demo.launch()
