from flask import Flask, request, jsonify
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from io import BytesIO
import base64
from PIL import Image

app = Flask(__name__)

# Load the model (run this only once when starting the server)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    image_file = request.files['image']
    prompt = request.form.get('prompt', 'high quality design variation')
    
    # Process the image
    input_image = Image.open(image_file).convert("RGB")
    
    # Generate variations
    result = pipe(
        prompt=prompt,
        image=input_image,
        strength=0.7,
        guidance_scale=7.5
    ).images[0]
    
    # Convert to base64 for JSON response
    buffered = BytesIO()
    result.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return jsonify({'result': img_str})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
