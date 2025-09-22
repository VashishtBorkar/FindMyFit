from pathlib import Path
import imageio.v3 as iio
import os
from dotenv import load_dotenv

# Path to your AVIF images
load_dotenv()
image_dir = Path(os.getenv("TEST_DATA_DIR"))

# Convert AVIF to PNG in the same folder
for file in image_dir.iterdir():
    try:
        # Read the AVIF image
        image = iio.imread(file)
        # Save as PNG
        png_file = image_dir / f"{file.stem}.png"
        iio.imwrite(png_file, image)
        print(f"Converted {file.name} → {png_file.name}")
        file.unlink()
    except Exception as e:
        print(f"Failed to convert {file.name}: {e}")

print("Conversion complete!")
