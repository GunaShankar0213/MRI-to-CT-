import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# AttentionUNet Model Definition (Custom Class)
class AttentionUNet:
    def __init__(self, img_rows=256, img_cols=256):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_shape = (self.img_rows, self.img_cols, 1)
        self.df = 64  # Downsampling filter size
        self.uf = 64  # Upsampling filter size

    def build_unet(self):
        def conv2d(layer_input, filters, dropout_rate=0, bn=False):
            d = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
            if bn:
                d = layers.BatchNormalization()(d)
            d = layers.Activation('relu')(d)

            d = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
            if bn:
                d = layers.BatchNormalization()(d)
            d = layers.Activation('relu')(d)

            if dropout_rate:
                d = layers.Dropout(dropout_rate)(d)

            return d

        def deconv2d(layer_input, filters, bn=False):
            u = layers.UpSampling2D((2, 2))(layer_input)
            u = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(u)
            if bn:
                u = layers.BatchNormalization()(u)
            u = layers.Activation('relu')(u)
            return u

        def attention_block(F_g, F_l, F_int, bn=False):
            g = layers.Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_g)
            if bn:
                g = layers.BatchNormalization()(g)
            x = layers.Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_l)
            if bn:
                x = layers.BatchNormalization()(x)

            psi = layers.Add()([g, x])
            psi = layers.Activation('relu')(psi)

            psi = layers.Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(psi)
            if bn:
                psi = layers.BatchNormalization()(psi)
            psi = layers.Activation('sigmoid')(psi)

            return layers.Multiply()([F_l, psi])

        inputs = layers.Input(shape=self.img_shape)

        # Contracting path (encoder)
        conv1 = conv2d(inputs, self.df)
        pool1 = layers.MaxPooling2D((2, 2))(conv1)

        conv2 = conv2d(pool1, self.df * 2, bn=True)
        pool2 = layers.MaxPooling2D((2, 2))(conv2)

        conv3 = conv2d(pool2, self.df * 4, bn=True)
        pool3 = layers.MaxPooling2D((2, 2))(conv3)

        conv4 = conv2d(pool3, self.df * 8, dropout_rate=0.5, bn=True)
        pool4 = layers.MaxPooling2D((2, 2))(conv4)

        conv5 = conv2d(pool4, self.df * 16, dropout_rate=0.5, bn=True)

        # Expanding path (decoder)
        up6 = deconv2d(conv5, self.uf * 8, bn=True)
        conv6 = attention_block(up6, conv4, self.uf * 8, bn=True)
        up6 = layers.Concatenate()([up6, conv6])
        conv6 = conv2d(up6, self.uf * 8)

        up7 = deconv2d(conv6, self.uf * 4, bn=True)
        conv7 = attention_block(up7, conv3, self.uf * 4, bn=True)
        up7 = layers.Concatenate()([up7, conv7])
        conv7 = conv2d(up7, self.uf * 4)

        up8 = deconv2d(conv7, self.uf * 2, bn=True)
        conv8 = attention_block(up8, conv2, self.uf * 2, bn=True)
        up8 = layers.Concatenate()([up8, conv8])
        conv8 = conv2d(up8, self.uf * 2)

        up9 = deconv2d(conv8, self.uf, bn=True)
        conv9 = attention_block(up9, conv1, self.uf, bn=True)
        up9 = layers.Concatenate()([up9, conv9])
        conv9 = conv2d(up9, self.uf)

        # Output layer
        outputs = layers.Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=outputs)

        return model

# Create a method to load the model with custom objects
def custom_load_model():
    # Instantiate the AttentionUNet class and return a model
    a = AttentionUNet()
    model = a.build_unet()
    model.load_weights(r'Model\attention_unet_model.h5')
    return model

# Function to preprocess the MRI image
def preprocess_image(image_path):
    mri_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if mri_image is None:
        raise ValueError(f"Error: Image not found or unable to load from path: {image_path}")

    # Resize to match model input size (256, 256)
    mri_image_resized = cv2.resize(mri_image, (256, 256))

    # Normalize the image
    mri_image_normalized = mri_image_resized / 255.0

    # Add batch and channel dimensions
    mri_image_input = np.expand_dims(mri_image_normalized, axis=(0, -1))
    return mri_image_input

# Streamlit UI for file upload and prediction
st.title("MRI to CT Image Conversion Using Attention U-Net")

st.write("Upload an MRI image to predict the corresponding CT image:")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Save the uploaded file temporarily
    with open("uploaded_mri_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the model with the custom AttentionUNet
    model = custom_load_model()

    # Preprocess the MRI image
    try:
        mri_image_input = preprocess_image("uploaded_mri_image.jpg")

        # Predict the corresponding CT image
        predicted_ct_image = model.predict(mri_image_input)

        # Post-process the predicted CT image
        predicted_ct_image = predicted_ct_image[0, :, :, 0]  # Remove batch and channel dims
        predicted_ct_image = (predicted_ct_image * 255).astype(np.uint8)  # Denormalize

        # Show the predicted CT image
        st.image(predicted_ct_image, caption="Predicted CT Image", use_column_width=True)

        # Optionally, you can save the predicted image
        cv2.imwrite("predicted_ct_image.png", predicted_ct_image)

        st.success("Prediction complete! CT image saved as 'predicted_ct_image.png'.")

    except ValueError as e:
        st.error(f"Error: {e}")
