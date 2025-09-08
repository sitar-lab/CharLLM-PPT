# stack_pdfs_true_vertical.py

import sys
from pypdf import PdfReader, PdfWriter, Transformation

def stack_first_pages_vertically(pdf_files, output_file):
    # Read the first page of each PDF
    pages = []
    widths = []
    heights = []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        page = reader.pages[0]
        w = float(page.mediabox.width)
        h = float(page.mediabox.height)
        pages.append(page)
        widths.append(w)
        heights.append(h)

    # Create a blank page tall enough to stack all
    max_width = max(widths)
    total_height = sum(heights)
    writer = PdfWriter()
    # Create a blank page
    from pypdf import PageObject
    blank = PageObject.create_blank_page(width=max_width, height=total_height)

    from copy import deepcopy
    y_offset = total_height
    for orig_page, h in zip(pages, heights):
        y_offset -= h
        page = deepcopy(orig_page)
        # Use merge_transformed_page if available
        try:
            blank.merge_transformed_page(page, Transformation().translate(tx=0, ty=y_offset))
        except AttributeError:
            # Fallback to manual transformation
            page.add_transformation(Transformation().translate(tx=0, ty=y_offset))
            blank.merge_page(page)

    writer.add_page(blank)
    with open(output_file, "wb") as f_out:
        writer.write(f_out)
    print(f"Saved stacked PDF as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python stack_pdfs_true_vertical.py file1.pdf file2.pdf file3.pdf output.pdf")
        sys.exit(1)
    pdf_files = sys.argv[1:3]
    output_file = sys.argv[3]
    stack_first_pages_vertically(pdf_files, output_file)