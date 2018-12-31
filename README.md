# Final Project for the Computer Science Foundation Class

## Group members: Yupeng Yang and Deen Huang

### Description:

 This Final Project is for our Computer Science Foundation class.

 This project focuses on the plant disease recognition by using Convolutional Neural Network model

 We build this publicly available repository on GitHub and our project team uses git for version control.

### Problem Statement & Dataset Description
 The diagnosis of pests and diseases is essential for agricultural production. We designed algorithms and models to recognize species and diseases in the crop leaves by using Convolutional Neural Network. There is a total of 31,147 images of diseased and healthy plants in our training dataset. These images span 10 original species of plants. Each set of images, including the training, validation, and testing, span 61 different species diseases of plants. Original images have various 2-dimensional sizes with different names.

### Components of our Repository:

* README.md

* AUTHORS.md

* LICENSE.md

* requirements.txt

* .gitignore

* setup.py

* Package folder: csf_modules:

> - plant_disease_data_process.py: Loading image data, processing data, resizing data and splitting data.

> - basic_cnn_model.py: this part includes build basic_cnn_model and train model.

> - deep_cnn_model.py: this part includes build deep_cnn_model and train model.

> - plot_loss_accuracy.py: this part plots the loss and accuracy values change with the increasing of epoches.

### Sample codes:
```python
def img_resize(imgpath, img_size):
    
    # resize the image to the specific size
    img = PIL.Image.open(imgpath)
    if (img.width > img.height):
        scale = float(img_size) / float(img.height)
        img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), img_size))).astype(np.float32)
    else:
        scale = float(img_size) / float(img.width)
        img = np.array(cv2.resize(np.array(img), (img_size, int(img.height * scale + 1)))).astype(np.float32)
        
    # crop the proper size and scale to [-1, 1]
    img = (img[
            (img.shape[0] - img_size) // 2:
            (img.shape[0] - img_size) // 2 + img_size,
            (img.shape[1] - img_size) // 2:
            (img.shape[1] - img_size) // 2 + img_size,
            :]-127)/128
            
    return img
```
#### Sample output (image after crop and resize):

![alt text][output-img]

[output-img]:resize.png "Output image after crop and resize"

### The image dataset can be found here:

[Plant Disease Recognition Dataset](https://drive.google.com/file/d/1x5yPRbF6_I-yS0zCS3zU26_3Ns9iXTr3/view?usp=sharing
)

### References:
* Francois Chollet, “Building powerful image classification models using very little data”, June 05, 2016
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
* Ziadé, Tarek. “Quick Start.” Quick Start - The Hitchhiker's Guide to Packaging 1.0 Documentation, the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/quickstart.html.
