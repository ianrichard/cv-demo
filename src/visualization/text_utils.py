import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class TextRenderer:
    """Class for rendering text with PIL for smooth fonts and positioning"""
    
    def __init__(self):
        """Initialize text renderer with font preloading"""
        # Try to load common fonts based on platform
        self.font_paths = [
            # macOS paths
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            # Linux paths
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            # Windows paths
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
        
        # Cache for loaded fonts
        self._fonts = {}
    
    def get_font(self, size=20):
        """Load or retrieve a cached font with the specified size"""
        if size in self._fonts:
            return self._fonts[size]
        
        # Try each font path
        for path in self.font_paths:
            try:
                font = ImageFont.truetype(path, size)
                self._fonts[size] = font
                return font
            except IOError:
                continue
        
        # Fall back to default
        self._fonts[size] = ImageFont.load_default()
        return self._fonts[size]
    
    def get_text_dimensions(self, draw, text, font):
        """Get width and height of text using textbbox"""
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    
    def draw_text_with_shadow(self, draw, text, position, font, 
                             text_color=(255, 255, 255),
                             shadow_color=(0, 0, 0), 
                             shadow_offset=2):
        """Draw text with shadow for better visibility on any background"""
        x, y = position
        draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_color)
        draw.text((x, y), text, font=font, fill=text_color)

    def render_status_text(self, frame, app):
        """Render status overlay with controls and mode indicators"""
        # Convert OpenCV BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_image)
        
        # Load fonts
        controls_font = self.get_font(size=20)
        status_font = self.get_font(size=22)
        
        # Get image dimensions for positioning
        width, height = pil_image.size
        
        # Bottom right positioning
        controls = "Controls: [Q]uit [C]amera [F]ace [O]bject"
        
        # Use textbbox for dimensions
        controls_width, controls_height = self.get_text_dimensions(draw, controls, controls_font)
        
        controls_x = width - controls_width - 10
        controls_y = height - controls_height - 10
        
        # Status texts at bottom right, stacked upwards
        face_status = "Face Detection: ON" if app.face_detection_enabled else "Face Detection: OFF"
        obj_status = "Object Detection: ON" if app.object_detection_enabled else "Object Detection: OFF"
        
        status_color_on = (0, 255, 0)  # Green for enabled
        status_color_off = (255, 0, 0)  # Red for disabled
        
        # Calculate text sizes
        face_width, face_height = self.get_text_dimensions(draw, face_status, status_font)
        obj_width, obj_height = self.get_text_dimensions(draw, obj_status, status_font)
        
        # Position status texts
        obj_x = width - obj_width - 10
        obj_y = height - controls_height - obj_height - 20
        
        face_x = width - face_width - 10
        face_y = obj_y - face_height - 10
        
        # Draw text elements with shadows
        self.draw_text_with_shadow(
            draw, controls, (controls_x, controls_y), 
            controls_font
        )
        
        self.draw_text_with_shadow(
            draw, obj_status, (obj_x, obj_y), 
            status_font,
            text_color=status_color_on if app.object_detection_enabled else status_color_off
        )
        
        self.draw_text_with_shadow(
            draw, face_status, (face_x, face_y), 
            status_font,
            text_color=status_color_on if app.face_detection_enabled else status_color_off
        )
        
        # Convert back to OpenCV format
        result_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result_frame

# Create a singleton instance
text_renderer = TextRenderer()

def add_status_text(frame, app):
    """Wrapper function for easier migration from existing code"""
    return text_renderer.render_status_text(frame, app)