## Image and Video Colorizer
Black and white image/video colorization with OpenCV.
https://image-colorizer-project.streamlit.app


### Demo sample of the app
<img width="3024" height="1806" alt="image" src="https://github.com/user-attachments/assets/b06ee645-a52a-4af2-b604-6680718382f5" />


### Overview
This project aims to provide a solution for image and video colorization using deep learning techniques. Using convolutional neural networks (CNNs) and modern web technologies, the project enables users to easily add color to grayscale images and videos. Image colorization is the process of taking an input grayscale (black and white) image and then producing an output colorized image that represents the semantic colors and tones of the input.

### Features

- Image colorization: Convert grayscale images to colorized versions.
- Video colorization: Extend image colorization to video content.
- User-friendly interface: Web-based interface for easy interaction and colorization.
- Real-time processing: Instant colorization of uploaded images and videos.
- Streamlit integration: Utilizes Streamlit for web application development and deployment.


### Technical Aspect

The colorization model is based on research by Richard Zhang, Phillip Isola, and Alexei A. Efros.

<img width="1291" height="353" alt="image" src="https://github.com/user-attachments/assets/89818302-bcf3-468f-8f63-da683df1f6df" />
https://richzhang.github.io/colorization/resources/images/net_diagram.jpg

- The technique we’ll be covering here today is from Zhang et al.’s 2016 ECCV paper, [Colorful Image Colorization](http://richzhang.github.io/colorization/). Developed at the University of California, Berkeley by Richard Zhang, Phillip Isola, and Alexei A. Efros.

- Previous approaches to black and white image colorization relied on manual human annotation and often produced    desaturated results that were not “believable” as true colorizations.

- Zhang et al. decided to attack the problem of image colorization by using Convolutional Neural Networks to  “hallucinate” what an input grayscale image would look like when colorized.

- To train the network Zhang et al. started with the [ImageNet dataset](http://image-net.org/) and converted all images from the RGB color space to the Lab color space.

- Similar to the RGB color space, the Lab color space has three channels. But unlike the RGB color space, Lab encodes color information differently:
  - The **L channel** encodes lightness intensity only
  - The **a channel** encodes green-red.
  - And the **b channel** encodes blue-yellow.

- As explained in the original paper, the authors, embraced the underlying uncertainty of the problem by posing it as a classification task using class-rebalancing at training time to increase the diversity of colors in the result. The Artificial Intelligent (AI) approach is implemented as a feed-forward pass in a CNN (“Convolutional Neural Network”) at test time and is trained on over a million color images.

### Installation And Run 
```bash
pip install -r requirements.txt
```
```bash
streamlit run app.py
```

### Tech Stack used

[<img target="_blank" src="https://github.com/user-attachments/assets/bfbabfc0-38cf-4d4c-9ee7-e19b0a8e83b6" width=180>](https://www.python.org)[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/OpenCV_Logo_with_text_svg_version.svg/730px-OpenCV_Logo_with_text_svg_version.svg.png" width=90>](https://opencv.org/)[<img target="_blank" src="https://miro.medium.com/max/4000/0*cSCGhssjeajRD3qs.png" width=200>](https://www.streamlit.io/)


### License
![Apache license](https://img.shields.io/badge/license-apache-blue?style=for-the-badge&logo=appveyor)



### Credits
- [“ Black and white image colorization with OpenCV and Deep Learning” by Dr. Adrian Rosebrok "](https://www.pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/) - This project wouldn't have been possible without these references.
- [The official publication of Zhang et al.](http://richzhang.github.io/colorization/)
