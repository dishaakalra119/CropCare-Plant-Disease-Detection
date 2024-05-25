import streamlit as st
import tensorflow as tf
import numpy as np

#Model Prediction Function

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

#SideBar
st.sidebar.title("Dashboard")
app_mode=st.sidebar.selectbox("Select Page", ["Home","About","Disease Recognition"])

if(app_mode=="Home"):
    import streamlit as st

    st.markdown("<h1 style='font-size:50px; text-align:center;'>CropCare</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:30px; text-align:center;'>Plant Disease Detection System</h1><br>", unsafe_allow_html=True)
    image_path="vadim-kaipov-8ZELrodSvTc-unsplash.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System!
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
                
    """)

## About Page


elif app_mode == "About":
    st.header("About")
    st.markdown("""
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset.
        This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure. A new directory containing 33 test images is created later for prediction purposes.
        #### Content
        1. train (70,295 images)
        2. test (33 images)
        3. validation (17,572 images)
        """)

    # Add the previous markdown text here

    st.markdown("""
        ### How CNN Works in Detail:
        #### Convolutional Neural Networks (CNNs):
        A CNN is a type of artificial intelligence specifically designed to process images. It can identify patterns and features in images to make decisions, such as detecting diseases in plant leaves.

        #### Layers of the CNN:
        """)
    st.image("cnnworking.PNG", use_column_width=True)  # Image about CNN
    st.markdown("<h1 style='font-size:20px; text-align:center;'>Diagram depicting the inner working of a Convolution Neural Network</h1><br>", unsafe_allow_html=True)

    st.markdown("""
        ##### Convolutional Layers:
        These layers act like a set of filters that scan over the image. Imagine looking at a photo with a magnifying glass to spot details. Each filter looks for specific features, such as edges, colors, or textures.
        For instance, one filter might detect the edges of a leaf, while another might highlight spots or discoloration.
                
        ##### Activation Function:
        After each convolutional layer, there's an activation function (like ReLU, which stands for Rectified Linear Unit) that helps the network decide which features are important. It keeps only the useful information and ignores the rest.
        
        ##### Pooling Layers:
        These layers reduce the size of the image while keeping the important information. It's like summarizing a long story by focusing on the key points. Pooling helps the CNN work faster and be less sensitive to the exact position of features in the image.

        ##### Fully Connected Layers:
        After several convolutional and pooling layers, the final part of the CNN is made up of fully connected layers. These layers are like the decision-making part of the network. They take the high-level features detected by the previous layers and use them to classify the image.
        For example, these layers will decide if the patterns found indicate a healthy leaf or a specific disease.

        #### Training the CNN:

        ##### Feeding Data:
        We feed the CNN thousands of images of plant leaves, each labeled as either healthy or diseased. The CNN uses these images to learn what healthy and diseased leaves look like.

        ##### Learning Process:
        The CNN adjusts its filters and decision rules based on the images. This process involves tweaking its settings to minimize errors. It's similar to how you might adjust your strategy in a game to improve your score.

        ##### Feedback Loop:
        The CNN continuously improves by comparing its predictions to the actual labels and making corrections. This process is repeated many times (through epochs) until the CNN can accurately identify the health of the leaves.

        ##### Making Predictions:

        Once the CNN is trained, it can analyze new images of plant leaves. When you upload a photo, the CNN processes it through its layers, identifying patterns and features it learned during training.

        ##### Final Decision:
        The fully connected layers at the end make a final decision about the image, such as classifying it as "healthy" or indicating a specific disease.

    
        #### Putting It All Together:

        ##### Image Upload:

        Users upload a picture of a plant leaf through a user-friendly web application.
        
        ##### Image Processing:
        The CNN processes the image, scanning for features and patterns.

                
        #### Prediction:
        Based on its training, the CNN predicts whether the leaf is healthy or diseased and displays the result to the user.
        By using a CNN, we leverage powerful AI technology to automatically and accurately diagnose plant diseases from images, helping farmers and gardeners maintain healthier crops
        """)
    st.markdown("[Learn more about CNN](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)")


    
    
## Disease Recognition Page

elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image=st.file_uploader("Choose an Image")
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)
    ## Prediction
    if(st.button("Predict")):
        with st.spinner("Please Wait"):
            st.write("Prediction:")
            result_index=model_prediction(test_image)
            ## Classes
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
            st.success("Model Prediction : {}".format(class_name[result_index]))

