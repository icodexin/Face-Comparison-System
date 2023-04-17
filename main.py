# Import the necessary libraries
import io
from PIL import Image

# Open the image file
image_path = "/Users/yangxin/Pictures/WechatIMG14.jpeg"  # Replace with your image file path
with open(image_path, "rb") as image_file:
    # Read the image file as binary
    image_bytes = image_file.read()

# Alternatively, if you already have the image data as bytes, you can skip the above step and directly use the bytes

# Convert the image data into a bytes-like object
image_bytes_obj = io.BytesIO(image_bytes)

# Open the image using PIL
image = Image.open(image_bytes_obj)

# Perform further operations with the image if needed
# ...

# Get the byte string of the image
image_byte_string = image_bytes_obj.getvalue()

# Print the byte string
print(image_byte_string)
