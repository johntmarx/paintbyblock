# Color-by-Number Generator

This tool converts images into color-by-number worksheets using AI-powered subject detection. It automatically identifies the main subject in an image, separates it from the background, and creates a numbered grid using Crayola colors.

## Features

- AI-powered subject detection using Meta's Segment Anything Model (SAM)
- Automatic color selection from Crayola 64 color palette
- Smart color separation between subject and background
- Customizable grid size and number of colors
- Generates both preview and printable worksheet
- Debug visualization of subject detection

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/color-by-number.git
cd color-by-number
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the SAM model checkpoint:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Usage

Basic usage:
```bash
python color.py -i input.jpg
```

This will generate:
- `preview.png`: A preview of the color-by-number image
- `worksheet.png`: The printable worksheet with numbers
- `mask_preview.png`: A debug visualization of subject detection

### Command Line Options

- `-i`, `--input`: Input image path (required)
- `-c`, `--colors`: Number of colors to use (default: 8)
- `-s`, `--squares`: Number of squares along the longest edge (default: 50)
- `--preview`: Custom preview output path
- `--worksheet`: Custom worksheet output path
- `--mask-preview`: Custom mask preview output path
- `--model`: Custom SAM model path

### Examples

Use more colors:
```bash
python color.py -i input.jpg --colors 20
```

Create a more detailed grid:
```bash
python color.py -i input.jpg -s 100
```

Specify custom output paths:
```bash
python color.py -i input.jpg --preview my_preview.png --worksheet my_worksheet.png
```

## How It Works

1. **Subject Detection**: Uses Meta's Segment Anything Model (SAM) to identify the main subject
2. **Color Selection**: 
   - Analyzes the image to find dominant colors
   - Maps colors to the closest matches from the Crayola palette
   - Ensures subject and background use different colors at boundaries
3. **Grid Generation**:
   - Creates a numbered grid maintaining the image's aspect ratio
   - Assigns numbers based on the selected colors
4. **Output Generation**:
   - Creates a full-color preview
   - Generates a printable worksheet with numbers
   - Produces a debug visualization of subject detection

## Notes

- The script requires a CUDA-capable GPU for optimal performance, but will fall back to CPU if none is available
- Processing time varies based on image size and grid complexity
- Subject detection works best with clear, well-lit subjects against contrasting backgrounds
- Best results are achieved with images where the subject is centered and well-defined

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU recommended (but not required)
- See requirements.txt for full package list

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
