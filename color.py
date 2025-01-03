import argparse
import math
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from collections import Counter
from segment_anything import SamPredictor, sam_model_registry

# Crayola 64 colors with their RGB values
COMMON_COLORS = {
    "Forest Green": (95, 167, 119),
    "Granny Smith Apple": (168, 228, 160),
    "Olive Green": (186, 184, 108),
    "Spring Green": (239, 255, 171),
    "Green Yellow": (240, 232, 145),
    "Yellow": (252, 232, 131),
    "Goldenrod": (255, 217, 47),
    "Bluetiful": (60, 105, 231),
    "Apricot": (253, 213, 177),
    "Peach": (255, 203, 164),
    "Yellow Orange": (255, 182, 83),
    "Orange": (255, 117, 56),
    "Red Orange": (255, 83, 73),
    "Scarlet": (242, 40, 71),
    "Melon": (255, 120, 112),
    "Brick Red": (196, 44, 46),
    "Red": (238, 32, 77),
    "Violet Red": (247, 83, 148),
    "Wild Strawberry": (255, 67, 164),
    "Magenta": (246, 100, 175),
    "Red Violet": (192, 68, 143),
    "Salmon": (255, 155, 170),
    "Tickle Me Pink": (252, 137, 172),
    "Carnation Pink": (255, 170, 204),
    "Mauvelous": (239, 152, 170),
    "Lavender": (252, 180, 213),
    "Orchid": (230, 168, 215),
    "Plum": (142, 69, 133),
    "Violet": (146, 110, 174),
    "Wisteria": (201, 160, 220),
    "Purple Mountains' Majesty": (157, 129, 186),
    "White": (255, 255, 255),
    "Silver": (205, 197, 194),
    "Timberwolf": (219, 215, 210),
    "Gray": (149, 145, 140),
    "Black": (0, 0, 0),
    "Gold": (231, 198, 151),
    "Macaroni And Cheese": (255, 189, 136),
    "Tan": (250, 167, 108),
    "Burnt Orange": (255, 127, 73),
    "Mahogany": (205, 74, 76),
    "Bittersweet": (253, 124, 110),
    "Chestnut": (188, 93, 88),
    "Burnt Sienna": (234, 126, 93),
    "Brown": (180, 103, 77),
    "Sepia": (165, 105, 79),
    "Raw Sienna": (214, 138, 89),
    "Tumbleweed": (222, 166, 129),
    "Blue Violet": (115, 102, 189),
    "Indigo": (79, 105, 198),
    "Blue": (31, 117, 254),
    "Cerulean": (29, 172, 214),
    "Cornflower": (154, 206, 235),
    "Pacific Blue": (28, 169, 201),
    "Cadet Blue": (176, 183, 198),
    "Blue Green": (13, 152, 186),
    "Periwinkle": (197, 208, 230),
    "Sky Blue": (128, 218, 235),
    "Turquoise Blue": (108, 218, 231),
    "Robin's Egg Blue": (31, 206, 203),
    "Asparagus": (135, 169, 107),
    "Green": (28, 172, 120),
    "Sea Green": (159, 226, 191),
    "Yellow Green": (197, 227, 132),
}

class ImageProcessor:
    def __init__(self, sam_checkpoint="sam_vit_h_4b8939.pth"):
        """Initialize with SAM model for segmentation."""
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)

    def generate_mask(self, image):
        """Generate mask for the main subject using SAM with enhanced contrast."""
        # Convert to CV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Create high contrast version for better segmentation
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Set image in predictor (use enhanced version for prediction)
        self.predictor.set_image(enhanced)
        
        # Generate multiple points for better subject detection
        height, width = cv_image.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Use multiple points in a grid pattern
        points = [
            [center_x, center_y],  # Center
            [center_x - width//4, center_y],  # Left
            [center_x + width//4, center_y],  # Right
            [center_x, center_y - height//4],  # Top
            [center_x, center_y + height//4],  # Bottom
        ]
        
        input_points = np.array(points)
        input_labels = np.ones(len(points))
        
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        # Select best mask
        best_mask = masks[np.argmax(scores)]
        
        # Clean up mask with morphological operations
        kernel = np.ones((5,5), np.uint8)
        best_mask = cv2.morphologyEx(best_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, kernel)
        
        return best_mask.astype(bool)

    def visualize_mask(self, image, mask, output_path="mask_preview.png"):
        """Create a visualization of the mask overlaid on the original image."""
        # Convert mask to RGB for visualization
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        mask_rgb[mask] = [0, 255, 0]  # Green for subject
        mask_rgb[~mask] = [255, 0, 0]  # Red for background
        
        # Create semi-transparent overlay
        overlay = Image.fromarray(mask_rgb).convert('RGBA')
        overlay.putalpha(128)  # 50% transparency
        
        # Overlay on original image
        original = Image.fromarray(np.array(image)).convert('RGBA')
        composite = Image.alpha_composite(original, overlay)
        composite.convert('RGB').save(output_path)

    def find_border_pixels(self, mask, dilation_size=3):
        """Find pixels at the border of the subject mask with adjustable thickness."""
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        border = dilated - eroded
        return border.astype(bool)

    def enhance_image(self, image):
        """Enhance image contrast and vibrancy."""
        img_array = np.array(image)
        
        # Stretch contrast
        p2, p98 = np.percentile(img_array, (2, 98), axis=(0, 1))
        img_array = np.clip(img_array, p2[None, None], p98[None, None])
        img_array = ((img_array - p2[None, None]) / (p98[None, None] - p2[None, None]) * 255).astype(np.uint8)
        
        # Convert back to PIL
        enhanced = Image.fromarray(img_array)
        
        # Apply enhancements
        enhancers = [
            (ImageEnhance.Contrast, 1.2),
            (ImageEnhance.Color, 1.3),
            (ImageEnhance.Sharpness, 1.2)
        ]
        
        for enhancer_class, factor in enhancers:
            enhanced = enhancer_class(enhanced).enhance(factor)
        
        return enhanced

    def get_dominant_colors(self, image, mask, border_mask, color_values, num_colors, num_background_colors=2):
        """Get dominant colors with enhanced separation for dark subjects."""
        pixels = np.array(image)
        mask = mask.astype(bool)
        
        # Get border pixels with extended area
        border_pixels = pixels[border_mask]
        border_color_indices = set()
        for pixel in border_pixels:
            closest = self.find_closest_color(pixel, color_values)
            border_color_indices.add(closest)
        
        # Separate subject and background
        subject_mask = mask & ~border_mask
        subject_pixels = pixels[subject_mask]
        background_pixels = pixels[~mask]
        
        def get_closest_colors(pixel_set, colors, count, exclude_indices=None, min_distance=30):
            if exclude_indices is None:
                exclude_indices = set()
            
            color_counts = Counter()
            for pixel in pixel_set:
                closest = self.find_closest_color(pixel, colors)
                if closest not in exclude_indices:
                    color_counts[closest] += 1
            
            # Filter colors to ensure minimum distance
            selected_colors = []
            for color_idx, _ in color_counts.most_common():
                if len(selected_colors) >= count:
                    break
                    
                # Check distance to previously selected colors
                color = np.array(colors[color_idx])
                if not any(np.sum((color - np.array(colors[sc]))**2) < min_distance**2 
                          for sc in selected_colors):
                    selected_colors.append(color_idx)
            
            return [(idx, color_counts[idx]) for idx in selected_colors]
        
        # Get subject colors with minimum distance enforcement
        subject_colors = get_closest_colors(subject_pixels, color_values, 
                                          num_colors - num_background_colors)
        subject_indices = [idx for idx, _ in subject_colors]
        subject_indices.extend(border_color_indices)
        subject_indices = list(set(subject_indices))[:num_colors - num_background_colors]
        
        # Get background colors, ensuring contrast with subject colors
        background_colors = get_closest_colors(background_pixels, color_values,
                                             num_background_colors,
                                             exclude_indices=set(subject_indices),
                                             min_distance=50)  # Increased minimum distance for background
        background_indices = [idx for idx, _ in background_colors]
        
        return subject_indices + background_indices

    def find_closest_color(self, pixel, colors):
        """Find closest color from the palette."""
        pixel = np.array(pixel)
        distances = [np.sum((pixel - np.array(color))**2) for color in colors]
        return np.argmin(distances)

    def create_color_by_number(self, input_path, num_colors=8, long_edge_squares=50,
                             output_preview="preview.png", output_worksheet="worksheet.png",
                             output_mask="mask_preview.png"):
        """Main function to create color-by-number image."""
        # Open and enhance image
        original = Image.open(input_path).convert("RGB")
        enhanced = self.enhance_image(original)
        
        # Generate mask and save visualization
        mask = self.generate_mask(enhanced)
        self.visualize_mask(enhanced, mask, output_mask)
        
        border_mask = self.find_border_pixels(mask, dilation_size=5)  # Increased border size
        
        # Resize maintaining aspect ratio
        width, height = enhanced.size
        aspect_ratio = width / height
        
        if width >= height:
            new_width = long_edge_squares
            new_height = round(new_width / aspect_ratio)
        else:
            new_height = long_edge_squares
            new_width = round(new_height * aspect_ratio)
        
        enhanced = enhanced.resize((new_width, new_height), Image.Resampling.LANCZOS)
        mask = Image.fromarray(mask).resize((new_width, new_height), Image.Resampling.NEAREST)
        border_mask = Image.fromarray(border_mask).resize((new_width, new_height), Image.Resampling.NEAREST)
        
        mask = np.array(mask)
        border_mask = np.array(border_mask)
        
        # Get color palette
        color_names = list(COMMON_COLORS.keys())
        color_values = list(COMMON_COLORS.values())
        
        # Get dominant colors ensuring border/background separation
        dominant_indices = self.get_dominant_colors(enhanced, mask, border_mask,
                                                  color_values, num_colors)
        selected_colors = [color_values[i] for i in dominant_indices]
        selected_names = [color_names[i] for i in dominant_indices]
        
        # Create quantized image
        quantized = Image.new('RGB', (new_width, new_height))
        color_indices = []
        
        # First pass: Process subject pixels
        for y in range(new_height):
            row_indices = [None] * new_width
            for x in range(new_width):
                if mask[y, x]:  # Subject pixel
                    pixel = enhanced.getpixel((x, y))
                    closest_idx = self.find_closest_color(pixel, selected_colors)
                    quantized.putpixel((x, y), selected_colors[closest_idx])
                    row_indices[x] = closest_idx
            color_indices.append(row_indices)
        
        # Second pass: Process background pixels with neighbor awareness
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for y in range(new_height):
            for x in range(new_width):
                if not mask[y, x]:  # Background pixel
                    pixel = enhanced.getpixel((x, y))
                    
                    # Get colors used by neighboring subject pixels
                    forbidden_colors = set()
                    for dy, dx in directions:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < new_height and 0 <= nx < new_width and 
                            mask[ny, nx] and color_indices[ny][nx] is not None):
                            forbidden_colors.add(color_indices[ny][nx])
                    
                    # Find closest allowable color
                    valid_colors = [(i, c) for i, c in enumerate(selected_colors) 
                                  if i not in forbidden_colors]
                    
                    if not valid_colors:  # If somehow no valid colors (shouldn't happen)
                        closest_idx = self.find_closest_color(pixel, selected_colors)
                    else:
                        # Find closest among valid colors
                        pixel_array = np.array(pixel)
                        distances = [np.sum((pixel_array - np.array(c))**2) 
                                   for _, c in valid_colors]
                        closest_idx = valid_colors[np.argmin(distances)][0]
                    
                    quantized.putpixel((x, y), selected_colors[closest_idx])
                    color_indices[y][x] = closest_idx
        
        # Create preview
        cell_size = 20
        preview = quantized.resize(
            (new_width * cell_size, new_height * cell_size),
            Image.Resampling.NEAREST
        )
        preview.save(output_preview)
        
        # Create worksheet
        self._create_worksheet(color_indices, selected_colors, selected_names,
                             new_width, new_height, output_worksheet)
        
        return preview, quantized

    def _create_worksheet(self, color_indices, colors, color_names,
                         width, height, output_path):
        """Create the numbered worksheet."""
        cell_size = 30
        worksheet_width = width * cell_size
        worksheet_height = height * cell_size
        
        # Calculate legend layout
        legend_cols = min(4, len(colors))
        legend_rows = (len(colors) + legend_cols - 1) // legend_cols
        legend_height = max(80, legend_rows * 30 + 20)
        
        # Create worksheet image
        worksheet = Image.new("RGB", (worksheet_width, worksheet_height + legend_height),
                            color="white")
        draw = ImageDraw.Draw(worksheet)
        
        # Set up font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw grid
        for y in range(height):
            for x in range(width):
                # Calculate positions
                top_left_x = x * cell_size
                top_left_y = y * cell_size
                bottom_right_x = top_left_x + cell_size
                bottom_right_y = top_left_y + cell_size
                
                # Draw cell
                draw.rectangle(
                    [(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)],
                    outline="black",
                    width=1
                )
                
                # Add number
                color_idx = color_indices[y][x]
                label = str(color_idx + 1)
                
                # Get text size
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Center text
                text_x = top_left_x + (cell_size - text_width) // 2
                text_y = top_left_y + (cell_size - text_height) // 2
                
                draw.text((text_x, text_y), label, fill="black", font=font)
        
        # Draw legend
        self._draw_legend(draw, colors, color_names, worksheet_height, 
                         worksheet_width, legend_cols, font)
        
        worksheet.save(output_path)

    def _draw_legend(self, draw, colors, color_names, top_offset, width,
                    cols, font):
        """Draw color legend in grid format."""
        cell_width = width // cols
        legend_top = top_offset + 10
        
        for i, (color, name) in enumerate(zip(colors, color_names)):
            row = i // cols
            col = i % cols
            
            # Calculate positions
            box_left = col * cell_width + 10
            box_top = legend_top + row * 30
            box_right = box_left + 20
            box_bottom = box_top + 20
            
            # Draw color swatch
            draw.rectangle(
                [(box_left, box_top), (box_right, box_bottom)],
                fill=color,
                outline="black"
            )
            
            # Add label
            label = f"{i + 1}: {name}"
            draw.text(
                (box_right + 10, box_top + 2),
                label,
                fill="black",
                font=font
            )

def main():
    parser = argparse.ArgumentParser(description="Create color-by-number image with subject segmentation")
    parser.add_argument("-i", "--input", required=True, help="Input image path")
    parser.add_argument("-c", "--colors", type=int, default=8, help="Number of colors (default: 8)")
    parser.add_argument("-s", "--squares", type=int, default=50, help="Squares on longest edge (default: 50)")
    parser.add_argument("--preview", default="preview.png", help="Preview output path")
    parser.add_argument("--worksheet", default="worksheet.png", help="Worksheet output path")
    parser.add_argument("--mask-preview", default="mask_preview.png", help="Mask preview output path")
    parser.add_argument("--model", default="sam_vit_h_4b8939.pth", help="SAM model path")
    
    args = parser.parse_args()
    
    processor = ImageProcessor(args.model)
    processor.create_color_by_number(
        args.input,
        num_colors=args.colors,
        long_edge_squares=args.squares,
        output_preview=args.preview,
        output_worksheet=args.worksheet,
        output_mask=args.mask_preview
    )

if __name__ == "__main__":
    main()
