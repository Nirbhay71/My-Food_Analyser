import os
import google.generativeai as genai
from PIL import Image
import gradio as gr

# --- Securely get your API key ---
# We will set this key in the Hugging Face settings later
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# --- This is our AI model from before ---
model = genai.GenerativeModel('gemini-1.5-flash')

# --- This is the same analysis function we created ---
# I've updated it to accept an image directly instead of a file path.
def analyze_image_for_food(pil_image):
    """
    Analyzes an image to identify food items and estimate quantities.
    """
    if not api_key:
        return "ERROR: Gemini API Key is not set. Please add it to the Hugging Face Space Secrets."

    try:
        # The prompt we perfected to get better estimates
        prompt = """
        You are an expert at analyzing images of food.
        Your task is to identify every food item in the image and provide its quantity.
        Follow these rules strictly:
        1.  List each item on a new line.
        2.  Use the format: food_item : quantity
        3.  If you can count an item exactly, provide the number (e.g., apple : 2).
        4.  If an item is in a large pile or group, provide an estimated number (e.g., "grapes : approx. 50+").
        5.  Do not identify non-food items.
        """
        response = model.generate_content([prompt, pil_image])
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

# --- This is the new part that creates the web interface ---
iface = gr.Interface(
    fn=analyze_image_for_food,
    inputs=gr.Image(type="pil", label="Upload a food image"),
    outputs=gr.Textbox(label="Analysis Result", lines=10),
    title="🍓 Food Analyzer AI 🥕",
    description="Upload a picture of food, and the AI will tell you what's in it and how much there is. Built with Gemini 1.5 Flash."
)

# --- This launches the web app ---
iface.launch()
