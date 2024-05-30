import os
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path
import gradio as gr

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
API_KEY = os.getenv("API_KEY")

# Set up the model with the API key
genai.configure(api_key=API_KEY)

# Set up the model
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 4000,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

def input_image_setup(file_loc):
    if not (img := Path(file_loc)).exists():
        raise FileNotFoundError(f"Could not find image: {img}")

    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": Path(file_loc).read_bytes()
            }
        ]
    return image_parts

def generate_gemini_response(input_prompt, text_input, image_loc):
    image_prompt = input_image_setup(image_loc)
    prompt_parts = [input_prompt + text_input, image_prompt[0]]
    response = model.generate_content(prompt_parts)
    return response.text

input_prompt = """You are an advanced AI model specializing in generating engaging and contextually relevant captions or post content for social media platforms. 
Given the context and an uploaded image, your task is to create a captivating caption or post content that resonates with the selected social media platform's audience.
Please analyze the provided image and the contextual description carefully. Use the following guidelines based on the social media platform specified:

1. **Instagram**: Focus on visually appealing, inspirational, and trendy content. Use relevant hashtags.
2. **Facebook**: Craft engaging and personal stories or updates. Aim for a friendly and conversational tone.
3. **Twitter**: Create concise, witty, and impactful tweets. Utilize popular hashtags and mentions.
4. **LinkedIn**: Develop professional and insightful posts. Emphasize expertise, industry relevance, and networking.
5. **Pinterest**: Write creative and informative descriptions. Highlight the aesthetic and practical aspects.

Output should be one after another caption/post (bulleted in case of more than 1)

The prompted message is: """

def upload_file(files, text_input, social_media_platform, num_captions=1):
    if not files:
        return None, "Image not uploaded"
    file_paths = [file.name for file in files]
    response = generate_gemini_response(input_prompt + f"\nPlatform: {social_media_platform}\n"+ f"Number of Captions/Posts: {num_captions}\n", text_input, file_paths[0])
    return file_paths[0], response

with gr.Blocks() as demo:
    header = gr.Label("Captionify: From Image to Engagement - Get the Perfect Caption!")
    text_input = gr.Textbox(label="Enter context for the image")
    social_media_input = gr.Dropdown(choices=["Instagram", "Facebook", "Twitter", "LinkedIn", "Pinterest"], label="Select social media platform")
    num_captions_input = gr.Number(label="Number of Captions/Posts")
    image_output = gr.Image()
    upload_button = gr.UploadButton("Click to upload an image", file_types=["image"], file_count="multiple")
    generate_button = gr.Button("Generate")
    
    file_output = gr.Textbox(label="Generated Caption/Post Content")
    
    def process_generate(files, text_input, social_media_input, num_captions_input):
        if not files:
            return None, "Image not uploaded"
        return upload_file(files, text_input, social_media_input, num_captions_input)
    
    upload_button.upload(fn=lambda files: files[0].name if files else None, inputs=[upload_button], outputs=image_output)
    generate_button.click(fn=process_generate, inputs=[upload_button, text_input, social_media_input, num_captions_input], outputs=[image_output, file_output])

demo.launch(debug=True)