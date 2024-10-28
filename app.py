# Imports
from transformers import pipeline
from langchain_community.llms import HuggingFaceHub, Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline
import os
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

# Image to text
def image_to_text(image_path):
    image_to_text = pipeline(
        "image-to-text", 
        model="Salesforce/blip-image-captioning-base",
        max_new_tokens=50  # Added this parameter to control generation length
    )
    text = image_to_text(image_path)[0]["generated_text"]
    return text

# LLM
def generate_story(scenario):
    # Initialize Ollama with a local model
    llm = Ollama(
        model="llama3.2",  # You can use other models like "mistral" or "codellama"
        temperature=0.7
    )
    
    # Create prompt template
    prompt = PromptTemplate(
        input_variables=["scenario"],
        template="""Write a creative and engaging short story (about 200 words) about this scenario: {scenario}. 
        Make it interesting and descriptive with a clear beginning, middle, and end."""
    )
    
    # Create chain using the new pattern
    chain = prompt | llm
    
    # Generate story using invoke
    story = chain.invoke({"scenario": scenario})
    return story

# Text to speech
def text_to_speech(text, output_path="output.mp3"):
    synthesizer = pipeline("text-to-speech", model="facebook/fastspeech2-en-ljspeech")
    speech = synthesizer(text)
    
    # Save the audio file
    with open(output_path, "wb") as f:
        f.write(speech["audio"])
    
    return output_path

# # Main execution
# if __name__ == "__main__":
#     # Example usage
#     image_path = "image.jpeg"
    
#     # Convert image to text
#     scenario = image_to_text(image_path)
#     print(f"Image description: {scenario}")
    
#     # Generate story from scenario
#     story = generate_story(scenario)
#     print(f"Generated story: {story}")
    
#     # # Convert story to speech
#     # audio_file = text_to_speech(story)
#     # print(f"Audio saved to: {audio_file}")

# Streamlit UI
st.set_page_config(page_title="Image to Story Generator", layout="wide")

# Title and description
st.title("🖼️ Image to Story Generator")
st.markdown("""
    Upload an image and watch as AI transforms it into a creative story!
    This app uses state-of-the-art AI to:
    1. Analyze your image
    2. Generate a description
    3. Create a unique story
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create columns for image and text
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        with st.spinner("Analyzing image..."):
            # Get image description
            image = Image.open(uploaded_file)
            scenario = image_to_text(image)
            
            # Display image description
            st.subheader("📝 Image Description")
            st.write(scenario)
    
    # Generate and display story
    st.subheader("📖 Generated Story")
    with st.spinner("Crafting your story..."):
        story = generate_story(scenario)
        
    # Display the story in a nice box
    st.markdown(f"""
    <div style="
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;">
        {story}
    </div>
    """, unsafe_allow_html=True)
    
    # Add a download button for the story
    st.download_button(
        label="Download Story",
        data=f"Image Description:\n{scenario}\n\nStory:\n{story}",
        file_name="generated_story.txt",
        mime="text/plain"
    )

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, Hugging Face, and Ollama")

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_text(image_path):
    image_to_text = pipeline(
        "image-to-text", 
        model="Salesforce/blip-image-captioning-base",
        max_new_tokens=10000  # Added this parameter to control generation length
    )
    text = image_to_text(image_path)[0]["generated_text"]
    return text

def generate_story(scenario):
    llm = Ollama(
        model="llama3.2",
        temperature=0.7
    )
    
    prompt = PromptTemplate(
        input_variables=["scenario"],
        template="""Write an interesting and funny to make people 
        laugh and creative and engaging short story (about 50 words) 
        about this scenario: {scenario}. """
    )
    
    chain = prompt | llm
    story = chain.invoke({"scenario": scenario})
    return story

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Generate description and story
        try:
            scenario = image_to_text(filepath)
            story = generate_story(scenario)
            
            return jsonify({
                'success': True,
                'image_path': f'/static/uploads/{filename}',
                'description': scenario,
                'story': story
            })
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)
