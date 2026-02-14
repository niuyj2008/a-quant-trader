import vtracer
import os

input_path = "/Users/niuyj/Downloads/workspace_Antigravity/Stock/system_logo.png"
output_path = "/Users/niuyj/Downloads/workspace_Antigravity/Stock/system_logo.svg"

if not os.path.exists(input_path):
    print(f"Error: Input file {input_path} not found.")
else:
    # Convert PNG to SVG
    # vtracer.convert(input_path, output_path, colormode='color', hierarchical='cutout', mode='spline', filter_speckle=4, color_precision=6, layer_difference=16, corner_threshold=60, length_threshold=4.0, max_iterations=10, splice_threshold=45, path_precision=3)
    # Using simpler call as per typical vtracer usage if defaults are desired
    vtracer.convert_image_to_svg_py(input_path, output_path)
    print(f"Successfully converted {input_path} to {output_path}")
