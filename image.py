# import streamlit as st
# import os
# import openai
# from PIL import Image
# import base64

# client = openai.OpenAI(
#     api_key=os.environ.get("SAMBANOVA_API_KEY","1a8f434f-2873-4d5d-bd86-836616720fbb"),
#     base_url="https://api.sambanova.ai/v1",
# )

# def get_image_insight(image_base64):
#     """Get insights from the uploaded image using the Sambanova API."""
#     try:
#         # API request using the new format
#         response = client.chat.completions.create(
#                     model='Llama-3.2-90B-Vision-Instruct',
#                     messages=[{"role": "user","content": f"What do you see in this image? Image: {image_base64}"}],
#                     temperature =  0.5,
#                     top_p = 0.1
#                 )

#         return response['choices'][0]['message']['content']
#     except Exception as e:
#         return f"Error: {str(e)}"

# def encode_image_to_base64(image_path):
#     """Convert an image to base64."""
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

# def main():
#     st.title("Image Insight Generator")
#     st.write("Upload an image to get insights!")

#     # Upload image
#     uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp", "gif", "tiff"])

#     if uploaded_file is not None:
#         # Display uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)

#         # Save the uploaded file temporarily
#         temp_file_path = "temp_image.jpg"
#         with open(temp_file_path, "wb") as temp_file:
#             temp_file.write(uploaded_file.getbuffer())

#         # Convert image to base64
#         image_base64 = encode_image_to_base64(temp_file_path)

#         # Get insights
#         st.write("Processing the image...")
#         insights = get_image_insight(image_base64)
        
#         # Display insights
#         st.subheader("Insights:")
#         st.write(insights)

#         # Clean up temporary file
#         os.remove(temp_file_path)

# if __name__ == "__main__":
#     main()


import streamlit as st
import os
import openai
from PIL import Image
import base64

client = openai.OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY", "1a8f434f-2873-4d5d-bd86-836616720fbb"),
    base_url="https://api.sambanova.ai/v1",
)

def get_image_insight(image_base64_chunk):
    """Get insights from a chunk of the image using the Sambanova API."""
    try:
        # API request using the new format
        response = client.chat.completions.create(
            model="Llama-3.2-90B-Vision-Instruct",
            messages=[{"role": "user", "content": f"Analyze this image part: {image_base64_chunk}"}],
            temperature=0.5,
            top_p=0.1
        )

        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

def encode_image_to_base64_chunks(image_path, chunk_size=65000):
    """Convert an image to base64 and split into chunks."""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Split into chunks of specified size
    return [encoded[i:i + chunk_size] for i in range(0, len(encoded), chunk_size)]

def main():
    st.title("Image Insight Generator")
    st.write("Upload an image to get insights!")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp", "gif", "tiff"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save the uploaded file temporarily
        temp_file_path = "temp_image.jpg"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        # Convert image to base64 chunks
        st.write("Preparing image for analysis...")
        image_chunks = encode_image_to_base64_chunks(temp_file_path)
        
        # Get insights from each chunk
        st.write("Processing the image in parts...")
        all_insights = []
        for idx, chunk in enumerate(image_chunks):
            st.write(f"Processing part {idx + 1} of {len(image_chunks)}...")
            insight = get_image_insight(chunk)
            all_insights.append(insight)
        
        # Aggregate insights
        st.subheader("Insights:")
        st.write("\n".join(all_insights))

        # Clean up temporary file
        os.remove(temp_file_path)

if __name__ == "__main__":
    main()
