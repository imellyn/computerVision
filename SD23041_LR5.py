import streamlit as st
import torch 
from torchvision import models, transforms
from PIL import Image 
import pandas as pd
import requests
import numpy as np

#Step 1: Configure the Streamlit page 
st.set_page_config(page_title = "Image Classification with PyTorch and Streamlit",
                   page_icon="",
                   layout="centered")

st.title("Simple Image Classification Web App")
st.write("Using **PyTorch ResNet-18 (pretrained on ImageNet)** + Streamlit")

#Step 3: set device to CPU
device = torch.device("cpu")

@st.cache_data
def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    r = requests.get(url)
    labels = r.text.strip().split("\n")
    return labels

#Step 4:  Load a pre-trained ResNet18 model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    return model


labels = load_imagenet_labels()
model = load_model()

#Step 5: Preprocess pipeline for ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

#Step 6: User interface to upload image
uploaded_file = st.file_uploader("Upload an image (jpg/png)",
                                 type = ["jpg", "jpeg", "png"])

if uploaded_file is not None:
    #Display Original Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Unploaded Image", use_column_width=True)

#Step 7: Convert to tensor 
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

#Step 8: Softmax and top-5
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    st.subheader("Top-5 Predictions:")
    top_classes = []
    top_probs = []
    for i in range(top5_prob.size(0)):
        class_name = labels[top5_catid[i]]
        prob = top5_prob[i].item()*100
        top_classes.append(class_name)
        top_probs.append(prob)
        st.write(f"{class_name}: {prob:.4f}")

#Step 9: Bar chart visualization
    df = pd.DataFrame({
        'Class': top_classes,
        'Probability': top_probs
    })
    st.bar_chart(df.set_index("Class"))