# concat_pdfs_horizontal.py

import sys
from pypdf import PdfReader, PdfWriter, Transformation, PageObject
from copy import deepcopy

def concat_first_pages_horizontally(pdf1, pdf2, output_file):
    # Read the first page of each PDF
    reader1 = PdfReader(pdf1)
    reader2 = PdfReader(pdf2)
    page1 = reader1.pages[0]
    page2 = reader2.pages[0]
    w1, h1 = float(page1.mediabox.width), float(page1.mediabox.height)
    w2, h2 = float(page2.mediabox.width), float(page2.mediabox.height)

    # Create a blank page wide enough to fit both, and tall enough for the tallest
    total_width = w1 + w2
    max_height = max(h1, h2)
    blank = PageObject.create_blank_page(width=total_width, height=max_height)

    # Place page1 at (0, 0), page2 at (w1, 0), both bottom-aligned
    page1_copy = deepcopy(page1)
    page2_copy = deepcopy(page2)

    # Use merge_transformed_page if available, else fallback
    try:
        blank.merge_transformed_page(page1_copy, Transformation().translate(tx=0, ty=0))
        blank.merge_transformed_page(page2_copy, Transformation().translate(tx=w1, ty=0))
    except AttributeError:
        page1_copy.add_transformation(Transformation().translate(tx=0, ty=0))
        blank.merge_page(page1_copy)
        page2_copy.add_transformation(Transformation().translate(tx=w1, ty=0))
        blank.merge_page(page2_copy)

    writer = PdfWriter()
    writer.add_page(blank)
    with open(output_file, "wb") as f_out:
        writer.write(f_out)
    print(f"Saved horizontally concatenated PDF as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python concat_pdfs_horizontal.py file1.pdf file2.pdf output.pdf")
        sys.exit(1)
    pdf1, pdf2, output_file = sys.argv[1:4]
    concat_first_pages_horizontally(pdf1, pdf2, output_file)