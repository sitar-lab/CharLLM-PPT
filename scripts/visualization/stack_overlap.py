import sys
from pypdf import PdfReader, PdfWriter, Transformation

def stack_first_pages_vertically(pdf_files, output_file, overlap=30):
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

    # Create a blank page tall enough to stack all, minus overlaps
    max_width = max(widths)
    total_height = sum(heights) - overlap * (len(pages) - 1)
    writer = PdfWriter()
    from pypdf import PageObject
    blank = PageObject.create_blank_page(width=max_width, height=total_height)

    from copy import deepcopy
    x_shift = 0  # horizontal shift for the top figure (in points)
    y_offset = 0
    num_pages = len(pages)
    for i, (orig_page, h) in enumerate(zip(reversed(pages), reversed(heights))):
        page = deepcopy(orig_page)
        # Move the top figure (i == 0) to the right by x_shift points
        x_offset = x_shift if i == 0 else 0
        try:
            blank.merge_transformed_page(page, Transformation().translate(tx=x_offset, ty=y_offset))
        except AttributeError:
            page.add_transformation(Transformation().translate(tx=x_offset, ty=y_offset))
            blank.merge_page(page)
        y_offset += h - overlap  # move down for overlap

    writer.add_page(blank)
    with open(output_file, "wb") as f_out:
        writer.write(f_out)
    print(f"Saved stacked PDF as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python stack_pdfs.py file1.pdf file2.pdf [file3.pdf ...] output.pdf [overlap_points]")
        sys.exit(1)
    # Support variable number of input PDFs and optional overlap argument
    if sys.argv[-1].isdigit():
        overlap = int(sys.argv[-1])
        pdf_files = sys.argv[1:-2]
        output_file = sys.argv[-2]
    else:
        overlap = 25  # default overlap
        pdf_files = sys.argv[1:-1]
        output_file = sys.argv[-1]
    stack_first_pages_vertically(pdf_files, output_file, overlap)