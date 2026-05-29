# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

from matplotlib import pyplot as plt
import torchvision 
import os
import re
from pathlib import Path

def visualize_images(images, nrow=4, normalize=True, figsize=(18, 18)):
    grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=normalize)

    # Convert the grid to a NumPy array and transpose the dimensions
    grid = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=figsize)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

def visualize_dataset_folder(folder_path, output_file):
    """
    Visualize the images in a dataset folder and its subfolders.

    Args:
        folder_path (str): The path to the dataset folder.

    Returns:
        None
    """
    # Get a list of all image files in the folder and its subfolders
    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                image_files.append(os.path.join(root, file))

    # Start the HTML content
    html_content = '<html>\n<head>\n<title>Dataset Visualization</title>\n</head>\n<body>\n<table border="1">\n'

    # Add each image to the table with 4 columns per row
    num_images = len(image_files)
    num_rows = (num_images + 3) // 4  # Calculate the number of rows needed
    for row in range(num_rows):
        html_content += '<tr>\n'
        for col in range(4):
            index = row * 4 + col
            if index < num_images:
                image_path = image_files[index]
                image_name = os.path.basename(image_path)
                rel_path = os.path.relpath(image_path, os.path.dirname(output_file))
                html_content += f'<td><img src="{rel_path}" alt="{image_name}"></td>\n'
            else:
                html_content += '<td></td>\n'
        html_content += '</tr>\n'

    # Close the HTML tags
    html_content += '</table>\n</body>\n</html>'

    #get only the folder name from the output file path
    output_folder = os.path.dirname(output_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  #create the folder for the ouput file if it does not exist

    # Write the HTML content to an output file
    with open(output_file, 'w') as file:
        file.write(html_content)

    print(f"HTML table saved to {output_file}")

def create_html_table_from_images(folder_list, output_file, labels=None):
    # Get a list of all image files in each folder
    image_files = []
    for folder in folder_list:
        folder_images = []
        for f in os.listdir(folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                # Extract the numeric part from the filename
                match = re.search(r'\d+', f)
                if match:
                    numeric_part = int(match.group())
                    folder_images.append((numeric_part, os.path.join(folder, f)))
        # Sort the images based on the numeric part
        folder_images.sort(key=lambda x: x[0])
        image_files.append([image_path for _, image_path in folder_images])

    # Start the HTML content
    html_content = '<html>\n<head>\n<title>Image Table</title>\n</head>\n<body>\n<table border="1">\n'

    # Add column headings
    html_content += '<tr>\n'
    if labels is not None:
        html_content += '<th>Labels</th>\n'
    for folder in folder_list:
        folder_name = os.path.basename(os.path.normpath(folder))
        html_content += f'<th>{folder_name}</th>\n'
    html_content += '</tr>\n'

    # Add each image to the table, with each folder occupying a separate column
    max_rows = max(len(folder_images) for folder_images in image_files)
    for row in range(max_rows):
        html_content += '<tr>\n'
        if labels is not None:
            label = labels[row] if row < len(labels) else ''
            html_content += f'<td>{label}</td>\n'
        for _, folder_images in enumerate(image_files):
            if row < len(folder_images):
                image_path = folder_images[row]
                image_name = os.path.basename(image_path)
                rel_path = os.path.relpath(image_path, os.path.dirname(output_file))
                html_content += f'<td><img src="{rel_path}" alt="{image_name}"></td>\n'
            else:
                html_content += '<td></td>\n'
        html_content += '</tr>\n'

    # Close the HTML tags
    html_content += '</table>\n</body>\n</html>'

    # Get the folder containing the output file
    output_folder = os.path.dirname(output_file)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create the folder for the output file if it does not exist

    # Write the HTML content to the output file
    with open(output_file, 'w') as file:
        file.write(html_content)

    print(f"HTML table saved to {output_file}")
    
def create_image_gallery(folder_path, output_html="gallery.html"):
    # Get all PNG files in the folder
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    # HTML template
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Gallery</title>
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
            }
            td {
                padding: 10px;
                text-align: center;
                width: 50%;
            }
            img {
                max-width: 100%;
                height: auto;
            }
            .filename {
                margin-top: 5px;
                font-family: Arial, sans-serif;
            }
        </style>
    </head>
    <body>
        <table>
    """
    
    # Add images to table, 2 per row
    for i in range(0, len(png_files), 2):
        html_content += "<tr>"
        
        # First column
        html_content += f"""
            <td>
                <img src="{png_files[i]}" alt="{png_files[i]}">
                <div class="filename">{png_files[i]}</div>
            </td>
        """
        
        # Second column (if exists)
        if i + 1 < len(png_files):
            html_content += f"""
                <td>
                    <img src="{png_files[i+1]}" alt="{png_files[i+1]}">
                    <div class="filename">{png_files[i+1]}</div>
                </td>
            """
        else:
            html_content += "<td></td>"  # Empty cell for odd number of images
            
        html_content += "</tr>"
    
    # Close HTML
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Write to file
    output_path = Path(folder_path) / output_html
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Gallery created at: {output_path}")