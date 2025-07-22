## Image Colorizer
Black and white image colorization with OpenCV.
https://image-colorizer-project.streamlit.app


### Demo sample of the app
<img width="1506" height="903" alt="Screenshot 2025-07-21 at 9 17 31 PM" src="https://github.com/user-attachments/assets/5ef430dd-e7fe-4c37-a163-d428ce587f30" />


### Overview
Image colorization is the process of taking an input grayscale (black and white) image and then producing an output colorized image that represents the semantic colors and tones of the input (for example, an ocean on a clear sunny day must be plausibly “blue” — it can’t be colored “hot pink” by the model).

## Motivation

When I learned linear algebra and came to know about how the machine inteprets pictures as tensors and concept of image segmentation. I remember there were some movies which was restored and picutured in theatre. I just came across Research papers of University of california in image colorization. And most iimportantly when I colorized photos of my Grandmother with gorgeous saree, that smile in my mother's face worth it.

### Technical Aspect
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

### Technologies Used

<img width="150" height="150" alt="6968821_preview" src="https://github.com/user-attachments/assets/bfbabfc0-38cf-4d4c-9ee7-e19b0a8e83b6" />

[<img target="_blank" src="https://github.com/user-attachments/assets/bfbabfc0-38cf-4d4c-9ee7-e19b0a8e83b6" width=150>](https://www.python.org)[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/OpenCV_Logo_with_text_svg_version.svg/730px-OpenCV_Logo_with_text_svg_version.svg.png" width=100>](https://opencv.org/)[<img target="_blank" src="https://miro.medium.com/max/4000/0*cSCGhssjeajRD3qs.png" width=200>](https://www.streamlit.io/)


### License
![Apache license](https://img.shields.io/badge/license-apache-blue?style=for-the-badge&logo=appveyor)



### Credits
- [“ Black and white image colorization with OpenCV and Deep Learning” by Dr. Adrian Rosebrok "](https://www.pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/) - This project wouldn't have been possible without these references.
- [The official publication of Zhang et al.](http://richzhang.github.io/colorization/)
