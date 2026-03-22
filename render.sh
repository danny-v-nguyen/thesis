#!/bin/bash
# 
# Call pandoc to render all markdown files to HTML.
# 

MD_DIR="reading"
OUTPUT_DIR="notes"
CURRENT_DIR=$(pwd)

# Find each markdown file in the target directory and render it to HTML
find "$MD_DIR" -type f -name "*.md" | while read -r md; do
    # Get the base name of the markdown file (without extension)
    base_name=$(basename "$md" .md)
    # Define the output HTML file path
    html_file="$CURRENT_DIR/$OUTPUT_DIR/$base_name.html"
    # Render the markdown file to HTML using pandoc
    pandoc "$md" -o "$html_file"
    echo "Rendered $md to $html_file"
done