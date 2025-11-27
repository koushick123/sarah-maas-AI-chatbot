from PIL import Image, ImageDraw, ImageFont

from PIL import Image, ImageDraw, ImageFont

def create_image_from_list_default_font(text_list, filename="output_default.jpg", bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    """
    Creates a JPEG image file from a list of text strings using the default font.
    """
    # Combine list items into a single string with newlines
    text_content = "\n".join(text_list)
    
    # Load the default bitmap font
    font = ImageFont.load_default()

    # We need a temporary/dummy image and draw object *first* to calculate the text size accurately
    # before creating the final image of the correct size.
    temp_img = Image.new('RGB', (1, 1)) # Small dummy image
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Calculate the bounding box of the text using the dummy draw object
    # This determines the required width and height for the actual image
    bbox = temp_draw.multiline_textbbox((0, 0), text_content, font=font)
    
    # Extract width and height from the bounding box
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Define padding
    padding = 20
    final_width = text_width + (padding * 2)
    final_height = text_height + (padding * 2)

    # Create the actual, correctly-sized image in RGB mode (JPEG does not support transparency)
    img = Image.new('RGB', (final_width, final_height), color=bg_color)
    
    # Now define the actual draw object associated with the final image
    draw = ImageDraw.Draw(img)

    # Draw the text onto the image with padding
    draw.multiline_text((padding, padding), text_content, font=font, fill=text_color, align="left")

    # Save the image as a JPEG file using default settings
    img.save(filename, 'JPEG')

    print(f"Successfully created image: {filename}")

# --- Example Usage ---
my_list = [
    "This text uses the default font.",
    "No specific font settings were applied.",
    "Pillow handles everything automatically.",
    "This is a longer line to test sizing calculations."
]


if __name__ == "__main__":
    create_image_from_list_default_font(my_list)

# You can also customize the settings:
# create_image_from_text_list(my_text_list, filename="custom_text_image.jpg", 
#                             font_path="C:/Windows/Fonts/times.ttf", font_size=36, 
#                             text_color=(255, 0, 0), background_color=(255, 255, 0))