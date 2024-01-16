import gradio as gr
from fastai.vision.all import *
from PIL import Image

# Load the trained model
learn = load_learner('dinosaur_classifier.pkl')

# Define a function that takes an image as input and returns the classification result
def classify_image(img):
    # Convert the image to a PIL Image if it's not already
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img, 'RGB')
    # Make a prediction
    pred, _, probs = learn.predict(img)
    # Get the top 3 predictions with their probabilities
    return {str(pred): float(max(probs))}

# Gradio interface components
image_input = gr.Image(label="Upload an image")
label_output = gr.Label(num_top_classes=3)

# Examples to be shown in the interface for quick testing
# Ensure these paths are correct relative to the working directory of the Gradio app
examples = ['dinosaur1.jpg', 'dinosaur2.jpg', 'dinosaur3.jpg', 'dinosaur4.jpg']

# Creating the Gradio interface
intf = gr.Interface(
    fn=classify_image,
    inputs=image_input,
    outputs=label_output,
    examples=examples,
    title="Dinosaur Species Classifier",
    description="Upload an image of a dinosaur, and the model will predict its species."
)

# Launch the Gradio app
intf.launch()
