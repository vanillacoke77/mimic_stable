import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate_lung_xray(
    prompt="High-resolution X-ray image of human lungs, medical imaging, realistic anatomy, grayscale",
    negative_prompt="blurry, low quality, distorted, color, text, watermark",
    num_images=1,
    guidance_scale=7.5,
    num_inference_steps=50,
    seed=None
):
    """
    Generate synthetic X-ray images of lungs using Stable Diffusion.
    
    Args:
        prompt (str): Text description of the desired image
        negative_prompt (str): What to avoid in the image
        num_images (int): Number of images to generate
        guidance_scale (float): How closely to follow the prompt
        num_inference_steps (int): More steps = better quality but slower
        seed (int): Random seed for reproducibility
        
    Returns:
        List of PIL.Image objects
    """
    # Load the pretrained Stable Diffusion model
    # Using a medical-specific model if available would be better
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,  # Disable safety checker for medical images
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate images
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images
    
    # Convert to grayscale to better simulate X-rays
    grayscale_images = []
    for img in images:
        grayscale = img.convert("L")
        grayscale_images.append(grayscale)
    
    return grayscale_images

# Example usage
if __name__ == "__main__":
    # Generate 2 lung X-ray images
    xray_images = generate_lung_xray(
        prompt="High-resolution posterior-anterior chest X-ray showing healthy human lungs, medical imaging, realistic anatomy, grayscale, high contrast",
        num_images=2,
        seed=42
    )
    
    # Save the generated images
    for i, img in enumerate(xray_images):
        img.save(f"generated_lung_xray_{i+1}.png")
        print(f"Saved generated_lung_xray_{i+1}.png")
    
    # Display the first image
    xray_images[0].show()
