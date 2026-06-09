"""Extract text from paper_vo.pdf for analysis."""
import sys

try:
    import PyPDF2
    with open("paper_vo.pdf", "rb") as f:
        reader = PyPDF2.PdfReader(f)
        print(f"Number of pages: {len(reader.pages)}")
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                print(f"\n=== PAGE {i+1} ===")
                print(text[:6000])
except ImportError:
    print("PyPDF2 not available, trying pdfplumber...")
    try:
        import pdfplumber
        with pdfplumber.open("paper_vo.pdf") as pdf:
            print(f"Number of pages: {len(pdf.pages)}")
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    print(f"\n=== PAGE {i+1} ===")
                    print(text[:6000])
    except ImportError:
        print("Neither PyPDF2 nor pdfplumber available.")
        print("Please install: pip install PyPDF2")
        sys.exit(1)