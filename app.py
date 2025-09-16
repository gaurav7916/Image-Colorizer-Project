# import the necessary packages
import numpy as np
import cv2
import os
import streamlit as st
from PIL import Image
import base64
import tempfile
import imutils
from imutils.video import VideoStream


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

# Function to colorize video
def video_colorizer(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    vf = cv2.VideoCapture(tfile.name) 
    # load model and cluster points      
    prototxt = "models_colorization_deploy_v2.prototxt"
    model = "colorization_release_v2.caffemodel"
    points = "pts_in_hull.npy"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]   
    # stframe = st.empty()
    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = vf.get(cv2.CAP_PROP_FPS) or 24
    width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
    temp_output_path = os.path.join(tempfile.gettempdir(), "colorized_output.mp4")
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    frame_count = 0
    while vf.isOpened():
        ret, frame = vf.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        scaled = frame.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

        # resize the Lab frame to 224x224 (the dimensions the colorization
        # network accepts), split channels, extract the 'L' channel, and
        # then perform mean centering
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        # pass the L channel through the network which will *predict* the
        # 'a' and 'b' channel values
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

        # resize the predicted 'ab' volume to the same dimensions as our
        # input frame, then grab the 'L' channel from the *original* input
        # frame (not the resized one) and concatenate the original 'L'
        # channel with the predicted 'ab' channels
        ab = cv2.resize(ab, (frame.shape[1], frame.shape[0]))
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

        # convert the output frame from the Lab color space to RGB, clip
        # any values that fall outside the range [0, 1], and then convert
        # to an 8-bit unsigned integer ([0, 255] range)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")        
        #stframe.image(colorized)    
        out.write(colorized)
        frame_count += 1
        percent_complete = int((frame_count / total_frames) * 100)
        yield percent_complete, None
    vf.release()
    out.release()

    yield 100, temp_output_path   
 
            
activities = ["Image","Video","About"]
choice = st.sidebar.selectbox("Choose what you want to Colorize.",activities)
if choice == "Image":
    st.title("Colorize Black and White Image 🎨")
    st.markdown("<h5>This is an app to turn any black and white images to a colored image (supported image types are jpg, jpeg, png upto 6000*4000 pixels). The model uses a pre-trained deep learning model for colorizing black and white images.</h6>", unsafe_allow_html=True)
    file = st.sidebar.file_uploader("Please upload an image file (upto 6000*4000 pixels)", type=["jpg", "jpeg", "png"])

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
            st.subheader("Original Image")
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
elif choice == "Video":
    st.title("Colorize Black and White Video 🎥")   
    st.markdown("<h5>This is an app to turn any black and white videos to a colored video (supported video types are mp4, avi, mov). The model uses a pre-trained deep learning model for colorizing black and white videos.</h6>", unsafe_allow_html=True)
    video_file = st.sidebar.file_uploader("Please upload a video file", type=["mp4", "avi", "mov"])
    if video_file is None:
        st.text("You haven't uploaded a video file")
    elif video_file is not None:
        # Create two columns side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Video")
            st.video(video_file)
        with col2:
            # Colorize the video
            st.subheader("Final Colorized Video")
        
            # Use session_state to store the colorized video path
            if "colorized_video_path" not in st.session_state or st.session_state.get("last_video_filename") != video_file.name:
                # Only colorize if new video is uploaded
                st.session_state["last_video_filename"] = video_file.name
                with st.spinner("Rendering colorized video..."):
                    progress_placeholder = st.empty()
                    with progress_placeholder:
                        progress_bar = st.progress(0)
                    output_path = None

                    # Call generator-based colorizer to yield progress
                    for percent_complete, result_path in video_colorizer(video_file):
                        progress_bar.progress(percent_complete)
                        if percent_complete == 100:
                            output_path = result_path

                    # Remove the progress bar
                    progress_placeholder.empty()

                    # Save in session_state
                    st.session_state["colorized_video_path"] = output_path
            else:
                output_path = st.session_state["colorized_video_path"]
                
            #output_path = video_colorizer(video_file)
            if output_path:
                st.video(output_path)
                with open(output_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Video",
                        data=file,
                        file_name="colorized_video.mp4",
                        mime="video/mp4"
                    )
                if btn:
                    st.success("Video downloaded successfully!")

                # Optional cleanup
                if os.path.exists("colorized_video.mp4"):
                    os.remove("colorized_video.mp4") 

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

elif choice == "About":
    st.markdown("<h2>About the App</h2>", unsafe_allow_html=True)   
    st.markdown("This app uses a pre-trained deep learning model to colorize black and white images and videos. This project aims to provide a solution for image and video colorization using deep learning (CNN trained on millions of color images) to hallucinate colors for grayscale input. The model doesn’t output exact RGB values but predicts chrominance bins (a,b channels) conditioned on luminance (L). By embedding 313 quantized color points, it reduces colorization to a classification + rebalancing problem instead of direct regression (which tends to produce desaturated results). Finally, the grayscale input’s luminance is merged with the predicted ab channels to reconstruct a full-color image.", unsafe_allow_html=True)

    st.markdown(""" ### Features
    - **Image colorization**: Convert grayscale images to colorized versions.
    - **Video colorization**: Extend image colorization to video content.
    - **User-friendly interface**: Web-based interface for easy interaction and colorization.
    - **Real-time processing**: Instant colorization of uploaded images and videos.
    - **Streamlit integration**: Utilizes Streamlit for web application development and deployment.

    ### Technical Aspect
    It takes a grayscale (or RGB) image/video → extracts lightness (L) → predicts color distribution (ab) using a CNN → merges → converts back to RGB → outputs a colorized image/video.
    
    The colorization model is based on research by Richard Zhang, Phillip Isola, and Alexei A. Efros.

    ![Model Diagram](https://richzhang.github.io/colorization/resources/images/net_diagram.jpg)

    Previous approaches to black and white image colorization relied on manual human annotation and often produced desaturated results that were not “believable” as true colorizations. Zhang et al. approached image colorization using Convolutional Neural Networks (CNNs) to “hallucinate” what an input grayscale image would look like when colorized.

    - Trained on ImageNet dataset.
    - Converted RGB to Lab color space:
        - **L** channel: Lightness
        - **a** channel: Green-Red
        - **b** channel: Blue-Yellow
    - Tackled the uncertainty in color prediction by reframing it as a classification task using class-rebalancing to produce diverse color results.

    The AI model performs colorization via a feed-forward pass through a CNN at test time, trained on over a million color images.

    <h5>Developed by: Gaurav Gupta</h5>
    """, unsafe_allow_html=True)
