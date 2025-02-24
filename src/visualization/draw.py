import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_rounded_box(image, x, y, w, h, radius, color, thickness):
    """
    Draws a rounded rectangle on the image with improved anti-aliasing.
    """
    # Create a separate image for the rounded rectangle to enable better anti-aliasing
    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    if thickness < 0:
        # For filled rectangles
        # First draw the filled rectangle without corners
        cv2.rectangle(mask, (x + radius, y), (x + w - radius, y + h), color, -1)
        cv2.rectangle(mask, (x, y + radius), (x + w, y + h - radius), color, -1)
        
        # Now draw the filled corners with higher-quality anti-aliasing
        corners = [
            ((x + radius, y + radius), 180),
            ((x + w - radius, y + radius), 270),
            ((x + radius, y + h - radius), 90),
            ((x + w - radius, y + h - radius), 0)
        ]
        
        for (center, angle) in corners:
            cv2.ellipse(mask, center, (radius, radius), angle, 0, 90, color, -1, cv2.LINE_AA)
    else:
        # For outlined rectangles with improved anti-aliasing
        cv2.line(mask, (x + radius, y), (x + w - radius, y), color, thickness, cv2.LINE_AA)
        cv2.line(mask, (x + radius, y + h), (x + w - radius, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(mask, (x, y + radius), (x, y + h - radius), color, thickness, cv2.LINE_AA)
        cv2.line(mask, (x + w, y + radius), (x + w, y + h - radius), color, thickness, cv2.LINE_AA)
        
        # Draw the ellipses with LINE_AA for smooth anti-aliasing
        cv2.ellipse(mask, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(mask, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(mask, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(mask, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
    
    # Blend the mask with the original image
    cv2.addWeighted(mask, 1, image, 1, 0, image)

def draw_bounding_boxes(image, boxes, class_ids, confidences, classes):
    """
    Draws bounding boxes and labels on the input image.
    """
    overlay = image.copy()
    background = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(len(boxes)):
        box = boxes[i]
        x, y, w, h = box
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        
        # Draw rounded rectangle in the background image
        draw_rounded_box(background, 
                        x + 4, y + 4,  # margin offset
                        w - 8, h - 8,  # size reduction for margin
                        radius=15, 
                        color=(144, 238, 144), 
                        thickness=-1)  # -1 for filled rectangle
        
        # Draw rounded rectangle border with increased thickness for smoother appearance
        draw_rounded_box(image, x, y, w, h, radius=15, color=(0, 200, 0), thickness=3)
        
        # Draw text using PIL with reduced offset
        image = draw_text_with_pil(
            image,
            label,
            (x + 18, y + 12),
            font_size=24,
            text_color=(255, 255, 255),
            bg_color=(0, 200, 0),
            corner_radius=8
        )
    
    # Blend the overlay with the original image
    alpha = 0.1
    cv2.addWeighted(background, alpha, image, 1, 0, image)
    
    return image

def draw_text_with_pil(cv2_im, text, position, font_size=32, 
                      text_color=(255, 255, 255), bg_color=(0, 200, 0), 
                      corner_radius=8):
    """Convert CV2 image to PIL, draw text, and convert back."""
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    
    # Create a drawing context
    draw = ImageDraw.Draw(pil_im)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFCompact.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Get text size
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Add non-uniform padding - less on top
    padding_top = 5      # Reduced top padding
    padding_bottom = 10
    padding_left = 10
    padding_right = 10
    
    x, y = position
    
    # Create a new image for the rounded rectangle background
    background = Image.new('RGBA', pil_im.size, (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(background)
    
    # Draw rounded rectangle with reduced top padding
    bg_draw.rounded_rectangle(
        [
            x - padding_left,
            y - padding_top,
            x + text_width + padding_right,
            y + text_height + padding_bottom
        ],
        radius=corner_radius,
        fill=bg_color
    )
    
    # Composite the background onto the main image
    pil_im = Image.alpha_composite(pil_im.convert('RGBA'), background)
    
    # Draw text
    draw = ImageDraw.Draw(pil_im)
    draw.text(position, text, font=font, fill=text_color)
    
    # Convert back to CV2 format
    result = cv2.cvtColor(np.array(pil_im.convert('RGB')), cv2.COLOR_RGB2BGR)
    return result