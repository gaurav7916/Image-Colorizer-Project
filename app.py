# import the necessary packages
import numpy as np
import cv2
import os
import streamlit as st
from PIL import Image
import base64

# Set page configuration
st.set_page_config(page_title="Colorize Black and White Image", page_icon="🎨", layout="wide")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )

add_bg_from_local('wallpaper_background.jpg')

def colorizer(pil_image):
    # Convert to RGB to ensure 3-channel format
    pil_image = pil_image.convert("RGB")
    img = np.array(pil_image)

    # load model and cluster points      
    prototxt = "models_colorization_deploy_v2.prototxt"
    model = "colorization_release_v2.caffemodel"
    points = "pts_in_hull.npy"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    # add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
   # scale the pixel intensities to the range [0, 1], and then convert the image from the BGR to Lab color space
    img = img.astype("float32") / 255.0
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # resize the Lab image to 224x224 (the dimensions the colorization network accepts), split channels, extract the 'L' channel, and then perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    # pass the L channel through the network which will *predict* the 'a' and 'b' channel values
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
     # grab the 'L' channel from the *original* input image (not the
    # resized one) and concatenate the original 'L' channel with the predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    # convert the output image from the Lab color space to RGB, then clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    # the current colorized image is represented as a floating point
    # data type in the range [0, 1] -- let's convert to an unsigned 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")
    return colorized


st.markdown("<h2 style='font-family: monospace, sans-serif; color:rgb(255, 255, 255); font-size: 30px'>Colorize Black and White Image 🎨</h2>", unsafe_allow_html=True)
st.markdown("<h5>This is an app to turn any black and white images to a colored image (supported image types are jpg, jpeg, png). The model uses a pre-trained deep learning model for colorizing black and white images.</h6>", unsafe_allow_html=True)

file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "jpeg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    img = np.array(image)
    
    # Colorize the image
    color = colorizer(image)
    
    # Create two columns side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("The Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Final Colorized Image")
        st.image(color, use_container_width=True)
        
        # Save and download the colorized image
        color = Image.fromarray(color)
        color.save("colorized_image.jpg")
        with open("colorized_image.jpg", "rb") as file:
            btn = st.download_button(
                label="Download Image",
                data=file,
                file_name="colorized_image.jpg",
                mime="image/jpeg"
            )
        # Save the colorized image
        if btn:
            st.success("Image downloaded successfully!")
        os.remove("colorized_image.jpg")  
        # Clean up the saved file after download
        if os.path.exists("colorized_image.jpg"):
            os.remove("colorized_image.jpg")

    # CSS for styling
    st.markdown("""
    <style>
        /* General styling */
        .stTextArea, .stTextInput, .stButton, .stMarkdown {
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
        
       /* Download button styling (targeting anchor tags inside stDownloadButton) */
        .stDownloadButton > button {
            font-family: 'Comic Sans MS', cursive, sans-serif;
            background: linear-gradient(135deg, #FF6B6B, #FFD93D);
            color: black;
            border: none;
            border-radius: 2em;
            padding: 0.7em 1.5em;
            font-size: 1.1rem;
            font-weight: bold;
            text-align: center;
            transition: background 0.3s, transform 0.2s ease;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }
        
        .stDownloadButton > button:hover {
            background-image: linear-gradient(to top, rgb(44, 44, 44), #252731);
            transform: scale(1.05);
        }

        /* Success message */
        .stSuccess {
            font-size: 1.5rem;
            text-align: center;
            margin-top: 1rem;
        }
        
        /* CSS variables and animations */
        :root {
            --bright-blue: rgb(0, 100, 255);
            --bright-green: rgb(0, 255, 0);
            --bright-red: rgb(255, 0, 0);
            --background: rgba(0, 0, 0, 0.8);
            --foreground: white;
            --border-size: 1px;
            --border-radius: 0.5em;
        }
        
        @property --border-angle-1 {
            syntax: "<angle>";
            inherits: true;
            initial-value: 0deg;
        }
        
        @property --border-angle-2 {
            syntax: "<angle>";
            inherits: true;
            initial-value: 90deg;
        }
        
        @property --border-angle-3 {
            syntax: "<angle>";
            inherits: true;
            initial-value: 180deg;
        }
        
        @keyframes rotateBackground {
            to { --border-angle-1: 360deg; }
        }
        
        @keyframes rotateBackground2 {
            to { --border-angle-2: -270deg; }
        }
        
        @keyframes rotateBackground3 {
            to { --border-angle-3: 540deg; }
        }
    </style>
    """, unsafe_allow_html=True)
