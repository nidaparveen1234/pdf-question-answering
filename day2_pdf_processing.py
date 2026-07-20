from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ============================================
# STEP 1: READ THE PDF
# ============================================

def read_pdf(file_path):
    """
    Opens a PDF and extracts all text from every page.
    Think of this like copy-pasting every page into one big string.
    """
    reader = PdfReader(file_path)
    
    all_text = ""
    
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:  # some pages might be empty or images
            all_text += text
            print(f"✅ Read page {page_number + 1}")
    
    return all_text

# ============================================
# STEP 2: SPLIT INTO CHUNKS
# ============================================

def split_into_chunks(text):
    """
    Why split? Because AI models can't process huge walls of text at once.
    They have a limit called 'context window'.
    
    So we split the text into smaller overlapping pieces.
    Overlap means the end of chunk 1 appears at the start of chunk 2.
    This makes sure we don't lose meaning at the boundaries.
    
    Think of it like cutting a long rope into pieces,
    but each piece shares a little bit with the next one.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # each chunk = max 500 characters
        chunk_overlap=50,     # 50 characters overlap between chunks
        length_function=len   # measure size by character count
    )
    
    chunks = splitter.split_text(text)
    return chunks

# ============================================
# STEP 3: RUN EVERYTHING
# ============================================

def main():
    print("📄 Starting PDF processing...\n")
    
    # read the pdf
    print("Step 1: Reading PDF...")
    raw_text = read_pdf("test.pdf")
    print(f"\n✅ Total characters read: {len(raw_text)}")
    print(f"Preview of first 200 characters:\n{raw_text[:200]}\n")
    
    # split into chunks
    print("Step 2: Splitting into chunks...")
    chunks = split_into_chunks(raw_text)
    print(f"\n✅ Total chunks created: {len(chunks)}")
    
    # show first 3 chunks so we can see what happened
    print("\nFirst 3 chunks preview:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)
    
    print("\n🎉 PDF processing complete! Ready for AI.")
    return chunks

# this runs main() when you run the file
if __name__ == "__main__":
    main()