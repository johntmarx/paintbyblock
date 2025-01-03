import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from segment_anything import SamPredictor, sam_model_registry
import io

# ------ Configuration / Constants ------
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


# ------ LOAD SAM AT STARTUP (caching recommended) ------
@st.cache_resource
def load_sam_model(checkpoint_path: str):
    """
    Cache the SAM model in memory for reuse, to avoid reloading it
    each time the user processes an image. Uses weights_only=True
    to address the PyTorch pickle warning if supported.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # If you have a newer segment_anything version, you can do:
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path, weights_only=True)
    except TypeError:
        # Fallback if older version doesn't accept weights_only
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)

    sam.to(device)
    return sam


# ------ Utility Functions ------
def enhance_image(image_pil: Image.Image) -> Image.Image:
    arr = np.array(image_pil)
    p2, p98 = np.percentile(arr, (2, 98), axis=(0, 1))
    # Avoid zero division
    p98 = np.maximum(p98, p2 + 1)
    arr = np.clip(arr, p2[None, None], p98[None, None])
    arr = ((arr - p2[None, None]) / (p98[None, None] - p2[None, None]) * 255).astype(np.uint8)
    out = Image.fromarray(arr)

    enhancers = [
        (ImageEnhance.Contrast, 1.2),
        (ImageEnhance.Color, 1.3),
        (ImageEnhance.Sharpness, 1.1),
    ]
    for enh_class, factor in enhancers:
        out = enh_class(out).enhance(factor)

    return out

def generate_mask_sam(image_pil: Image.Image, loaded_sam) -> np.ndarray:
    """
    Generate a single subject mask using SAM (center + quarter offsets).
    loaded_sam is the model from our cache.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SamPredictor(loaded_sam)

    # Convert PIL -> BGR for SAM
    cv_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    predictor.set_image(cv_image)

    height, width = cv_image.shape[:2]
    cx, cy = width // 2, height // 2

    points = np.array([
        [cx, cy],
        [cx - width//4, cy],
        [cx + width//4, cy],
        [cx, cy - height//4],
        [cx, cy + height//4],
    ])
    labels = np.ones(len(points), dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True
    )
    best_mask = masks[np.argmax(scores)]

    kernel = np.ones((5,5), np.uint8)
    best_mask = cv2.morphologyEx(best_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, kernel)

    return best_mask.astype(bool)

def visualize_mask(image_pil: Image.Image, mask_bool: np.ndarray) -> Image.Image:
    if mask_bool is None:
        return None
    h, w = mask_bool.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[ mask_bool] = [0, 255, 0]
    overlay[~mask_bool] = [255, 0, 0]
    overlay_img = Image.fromarray(overlay, mode="RGB").convert("RGBA")
    overlay_img.putalpha(128)  # semi-transparency

    base = image_pil.convert("RGBA")
    composite = Image.alpha_composite(base, overlay_img)
    return composite.convert("RGB")


def find_closest_color(pixel, color_vals):
    px = np.array(pixel, dtype=np.int32)
    dists = [np.sum((px - np.array(c, dtype=np.int32))**2) for c in color_vals]
    return np.argmin(dists)


def find_border_pixels(mask_bool: np.ndarray, dilation_size=5) -> np.ndarray:
    import cv2
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1)
    eroded  = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=1)
    border = dilated - eroded
    return border.astype(bool)


def get_dominant_colors(
    image_pil: Image.Image,
    subject_mask: np.ndarray,
    border_mask: np.ndarray,
    palette_dict: dict,
    num_colors=8,
    num_background_colors=2
):
    color_names = list(palette_dict.keys())
    color_vals  = list(palette_dict.values())

    arr = np.array(image_pil)
    if subject_mask is None:
        subject_mask = np.zeros((arr.shape[0], arr.shape[1]), dtype=bool)
    if border_mask is None:
        border_mask = np.zeros_like(subject_mask, dtype=bool)

    # Check border usage
    border_px = arr[border_mask]
    border_used = set()
    for px in border_px:
        cidx = find_closest_color(px, color_vals)
        border_used.add(cidx)

    subj_px = arr[ subject_mask & ~border_mask]
    back_px = arr[~subject_mask]

    from collections import Counter

    def cluster_pixels(px_array, color_vals, count, exclude=None, min_distance=30):
        if exclude is None:
            exclude = set()
        ccounts = Counter()
        for p in px_array:
            idx = find_closest_color(p, color_vals)
            if idx not in exclude:
                ccounts[idx] += 1
        selected = []
        for idx, _freq in ccounts.most_common():
            if len(selected) >= count:
                break
            candidate = np.array(color_vals[idx], dtype=np.int32)
            too_close = False
            for sidx in selected:
                dist_sq = np.sum((candidate - np.array(color_vals[sidx], dtype=np.int32))**2)
                if dist_sq < min_distance**2:
                    too_close = True
                    break
            if not too_close:
                selected.append(idx)
        return selected

    scount = max(1, num_colors - num_background_colors)
    subj_indices = cluster_pixels(subj_px, color_vals, scount)
    # also keep border-used
    subj_set = set(subj_indices) | border_used
    subj_indices = list(subj_set)[:scount]
    bg_indices = cluster_pixels(back_px, color_vals, num_background_colors,
                                exclude=set(subj_indices), min_distance=50)
    final_indices = subj_indices + bg_indices
    selected_colors = [color_vals[i] for i in final_indices]
    selected_names  = [color_names[i] for i in final_indices]
    return selected_colors, selected_names


def create_color_by_number(
    image_pil: Image.Image,
    subject_mask: np.ndarray,
    palette_dict: dict,
    num_colors=8,
    squares_on_edge=50
):
    # Enhance
    enhanced = enhance_image(image_pil)

    # Border
    if subject_mask is not None:
        border_mask = find_border_pixels(subject_mask, dilation_size=5)
    else:
        border_mask = None

    # Resize
    w, h = enhanced.size
    aspect = w/h
    if w >= h:
        new_w = squares_on_edge
        new_h = int(round(new_w/aspect))
    else:
        new_h = squares_on_edge
        new_w = int(round(new_h*aspect))

    # Convert subject mask
    if subject_mask is not None:
        mask_pil = Image.fromarray(subject_mask).resize((new_w, new_h), Image.Resampling.NEAREST)
        mask_bool = np.array(mask_pil, dtype=bool)
    else:
        mask_bool = np.zeros((h, w), dtype=bool)

    if border_mask is not None:
        border_pil = Image.fromarray(border_mask).resize((new_w, new_h), Image.Resampling.NEAREST)
        border_bool = np.array(border_pil, dtype=bool)
    else:
        border_bool = None

    enh_resized = enhanced.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Pick palette
    selected_colors, selected_names = get_dominant_colors(
        enh_resized, mask_bool, border_bool,
        palette_dict, num_colors=num_colors
    )

    # Quantize
    arr_enh = np.array(enh_resized)
    quantized = Image.new("RGB", (new_w, new_h))
    color_indices = []

    # subject first
    for row in range(new_h):
        row_idxs = [None]*new_w
        for col in range(new_w):
            if mask_bool[row, col]:
                px = arr_enh[row, col]
                cidx = find_closest_color(px, selected_colors)
                quantized.putpixel((col, row), selected_colors[cidx])
                row_idxs[col] = cidx
        color_indices.append(row_idxs)

    # background
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for y in range(new_h):
        for x in range(new_w):
            if not mask_bool[y, x]:
                px = arr_enh[y, x]
                forbidden = set()
                for dy, dx in neighbors:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < new_h and 0 <= nx < new_w:
                        if mask_bool[ny, nx] and color_indices[ny][nx] is not None:
                            forbidden.add(color_indices[ny][nx])
                valid = [(i,c) for i,c in enumerate(selected_colors) if i not in forbidden]
                if not valid:
                    cidx = find_closest_color(px, selected_colors)
                else:
                    px_i = np.array(px, dtype=np.int32)
                    dists = [np.sum((px_i - np.array(c,dtype=np.int32))**2) for _,c in valid]
                    best_local = np.argmin(dists)
                    cidx = valid[best_local][0]
                quantized.putpixel((x,y), selected_colors[cidx])
                color_indices[y][x] = cidx

    # Upsample preview
    cell_size = 20
    preview_img = quantized.resize((new_w*cell_size, new_h*cell_size), Image.Resampling.NEAREST)

    # Worksheet
    worksheet_img = create_worksheet(color_indices, selected_colors, selected_names)

    return worksheet_img, preview_img

def create_worksheet(color_indices, palette_vals, palette_names):
    height = len(color_indices)
    width  = len(color_indices[0]) if height>0 else 0
    cell_size = 30
    ws_w = width * cell_size
    ws_h = height* cell_size

    legend_cols = min(4, len(palette_vals))
    legend_rows = (len(palette_vals)+legend_cols-1)//legend_cols
    legend_h = max(80, legend_rows*30+20)

    worksheet = Image.new("RGB", (ws_w, ws_h+legend_h), "white")
    draw = ImageDraw.Draw(worksheet)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for row in range(height):
        for col in range(width):
            x0 = col*cell_size
            y0 = row*cell_size
            x1 = x0+cell_size
            y1 = y0+cell_size
            draw.rectangle([x0,y0,x1,y1], outline="black", width=1)
            cidx = color_indices[row][col]
            label = str(cidx+1) if cidx is not None else "-"
            bbox = draw.textbbox((0,0), label, font=font)
            tw = bbox[2]-bbox[0]
            th = bbox[3]-bbox[1]
            tx = x0 + (cell_size - tw)//2
            ty = y0 + (cell_size - th)//2
            draw.text((tx,ty), label, fill="black", font=font)

    draw_legend(draw, palette_vals, palette_names, ws_h, ws_w, legend_cols, font)
    return worksheet

def draw_legend(draw, palette_vals, palette_names, top_offset, width, cols, font):
    cell_w = width//cols
    legend_top = top_offset+10
    for i,(cval,cname) in enumerate(zip(palette_vals, palette_names)):
        r = i//cols
        c = i%cols
        x0 = c*cell_w+10
        y0 = legend_top + r*30
        x1 = x0+20
        y1 = y0+20
        draw.rectangle([x0,y0,x1,y1], fill=cval, outline="black")
        label = f"{i+1}: {cname}"
        draw.text((x1+10,y0+2), label, fill="black", font=font)


# ------ STREAMLIT APP ------
def main():
    st.title("Color-by-Number Generator (Streamlit)")

    # Sidebar for user parameters
    sam_checkpoint = st.sidebar.text_input(
        "SAM Model Checkpoint",
        value="sam_vit_h_4b8939.pth",
        help="Path or filename for the SAM checkpoint."
    )
    num_colors = st.sidebar.slider("Number of colors", min_value=1, max_value=64, value=8)
    squares = st.sidebar.slider("Squares on longest edge", min_value=5, max_value=100, value=50)

    # Load the SAM model once at startup
    sam_model = load_sam_model(sam_checkpoint)

    uploaded_file = st.file_uploader("Upload an image (JPEG/PNG)", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        if st.button("Generate Color-by-Number"):
            with st.spinner("Segmenting and generating worksheets..."):
                # 1) Generate mask
                subject_mask = generate_mask_sam(input_image, sam_model)
                # 2) Create color-by-number
                worksheet_img, preview_img = create_color_by_number(
                    input_image, subject_mask, COMMON_COLORS,
                    num_colors=num_colors,
                    squares_on_edge=squares
                )
                # 3) Mask overlay
                mask_overlay = visualize_mask(input_image, subject_mask)

            # TABS for images
            tab1, tab2, tab3, tab4 = st.tabs(["Original", "Worksheet", "Preview", "Mask"])
            with tab1:
                st.image(input_image, caption="Original Image", use_container_width=True)
            with tab2:
                st.image(worksheet_img, caption="Worksheet", use_container_width=True)
            with tab3:
                st.image(preview_img, caption="Preview", use_container_width=True)
            with tab4:
                if mask_overlay is not None:
                    st.image(mask_overlay, caption="Mask Overlay", use_container_width=True)
                else:
                    st.write("No mask to display.")
        else:
            st.info("Click 'Generate Color-by-Number' to process the image.")
    else:
        st.warning("Please upload an image first.")

if __name__ == "__main__":
    main()
