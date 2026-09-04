import numpy as np
from PIL import Image, ImageDraw, ImageFont

out_dir = "paper/NIPS workshop/figures/panels"


# Load raw image arrays (RGBA or RGB)
def load_img(name):
    im = Image.open(f"{out_dir}/{name}").convert("RGB")
    return np.array(im)


gt_mag = load_img("gt_mag.png")
gt_pha = load_img("gt_pha.png")

euc_mag = load_img("euc_mag.png")
euc_mag_err = load_img("euc_mag_err.png")
euc_pha = load_img("euc_pha.png")
euc_pha_err = load_img("euc_pha_err.png")

cyl_mag = load_img("cyl_mag.png")
cyl_mag_err = load_img("cyl_mag_err.png")
cyl_pha = load_img("cyl_pha.png")
cyl_pha_err = load_img("cyl_pha_err.png")

H, W, C = gt_mag.shape
print(f"Slice image dimensions: H={H}, W={W}")

# Define exact pixel spacers (white = 255)
spacer_2px = np.full((H, 3, 3), 255, dtype=np.uint8)  # 3px tight gap
spacer_10px = np.full((H, 14, 3), 255, dtype=np.uint8)  # 14px mid gap
block_gap = np.full((H, 45, 3), 255, dtype=np.uint8)  # 45px block gap


# Helper to render text box as an RGB array of height H
def make_text_box(title, lines, width=480, title_color=(0, 0, 0), body_color=(50, 50, 50)):
    im = Image.new("RGB", (width, H), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    # Try loading a clean TTF or default font
    try:
        font_title = ImageFont.truetype(
            "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf", 52
        )
        font_body = ImageFont.truetype(
            "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf", 38
        )
        font_body_bold = ImageFont.truetype(
            "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf", 38
        )
    except:
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 52)
            font_body = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 38)
            font_body_bold = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 38)
        except:
            font_title = font_body = font_body_bold = ImageFont.load_default()

    # Draw Title
    draw.text((20, H // 2 - 90), title, fill=title_color, font=font_title)

    # Draw body lines
    y_off = H // 2 - 20
    for i, line in enumerate(lines):
        f = font_body_bold if "28.39" in line or "0.7624" in line or "0.2347" in line else font_body
        draw.text((20, y_off + i * 50), line, fill=body_color, font=f)

    return np.array(im)


# Build Block A
txt_a = make_text_box("(a) Ground Truth", ["Reference Target", "Full Complex Field"], width=460)
block_a = np.hstack([txt_a, gt_mag, spacer_2px, gt_pha])

# Build Block B
txt_b = make_text_box(
    "(b) Euclidean (R²)", ["PSNR: 25.35 dB", "SSIM: 0.6460", "CPE: 0.5261 rad"], width=460
)
block_b = np.hstack(
    [txt_b, euc_mag, spacer_2px, euc_pha, spacer_10px, euc_mag_err, spacer_2px, euc_pha_err]
)

# Build Block C
txt_c = make_text_box(
    "(c) Cylindrical (Ours)",
    ["PSNR: 28.39 dB", "SSIM: 0.7624", "CPE: 0.2347 rad"],
    width=520,
    title_color=(0, 68, 170),
    body_color=(0, 51, 136),
)
block_c = np.hstack(
    [txt_c, cyl_mag, spacer_2px, cyl_pha, spacer_10px, cyl_mag_err, spacer_2px, cyl_pha_err]
)

# Final horizontal composite
full_img = np.hstack([block_a, block_gap, block_b, block_gap, block_c])

# Save as clean uncompressed PNG
out_file = "paper/NIPS workshop/figures/figure2_results.png"
Image.fromarray(full_img).save(out_file, dpi=(300, 300))
print(f"Saved pixel-perfect composite image to {out_file}, shape: {full_img.shape}")
