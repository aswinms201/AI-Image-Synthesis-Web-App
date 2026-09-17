import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.set_page_config(page_title="Text-to-Image Generator", page_icon="🎨", layout="centered")

st.title("🎨 Stable Diffusion Image Generator")
st.write("Generate images from text prompts using Stable Diffusion v1.5")


@st.cache_resource(show_spinner="Loading model... this may take a minute the first time")
def load_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    return pipe


# Load once, cached across reruns/sessions
pipe = load_pipeline()

# --- Sidebar controls ---
st.sidebar.header("Generation settings")
height = st.sidebar.select_slider("Height", options=[256, 384, 512, 640, 768], value=512)
width = st.sidebar.select_slider("Width", options=[256, 384, 512, 640, 768], value=512)
num_inference_steps = st.sidebar.slider("Inference steps", min_value=10, max_value=100, value=50)
guidance_scale = st.sidebar.slider("Guidance scale", min_value=1.0, max_value=20.0, value=7.5, step=0.5)
seed = st.sidebar.number_input("Seed (-1 for random)", value=-1, step=1)

# --- Main input ---
prompt = st.text_input("Enter your prompt", value="a cat sitting on a chair")
negative_prompt = st.text_input("Negative prompt (optional)", value="")

generate_btn = st.button("Generate Image", type="primary")

if generate_btn:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):
            generator = None
            if seed != -1:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=device).manual_seed(int(seed))

            result = pipe(
                prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            image = result.images[0]

        st.image(image, caption=prompt, use_container_width=True)

        # Offer download
        import io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button(
            label="Download image",
            data=buf.getvalue(),
            file_name="generated_image.png",
            mime="image/png",
        )
