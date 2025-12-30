import pdfplumber

PDF_PATH = "data/raw/Rohan Resume MAIN.pdf"

def main():
    print("Opening PDF...")
    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        print("\n--- PAGE 1 TEXT PREVIEW ---\n")
        text = pdf.pages[0].extract_text()
        print(text)

if __name__ == "__main__":
    main()