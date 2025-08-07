#!/usr/bin/env python3
"""
Convert HTML notebook outputs to PDFs for validation against MATLAB examples.
"""
import os
import sys
import subprocess
from pathlib import Path
import argparse

def check_dependencies():
    """Check if required PDF conversion tools are installed."""
    tools = {
        'wkhtmltopdf': 'wkhtmltopdf --version',
        'weasyprint': 'python -c "import weasyprint"',
        'playwright': 'python -c "import playwright"'
    }
    
    available = {}
    for tool, check_cmd in tools.items():
        try:
            subprocess.run(check_cmd, shell=True, check=True, 
                         capture_output=True, text=True)
            available[tool] = True
        except subprocess.CalledProcessError:
            available[tool] = False
    
    return available

def install_weasyprint():
    """Install weasyprint if not available."""
    print("Installing weasyprint...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "weasyprint"], 
                      check=True)
        print("✓ weasyprint installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install weasyprint")
        return False

def convert_with_weasyprint(html_path, pdf_path):
    """Convert HTML to PDF using weasyprint."""
    try:
        import weasyprint
        print(f"Converting {html_path.name} to PDF...")
        weasyprint.HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return True
    except Exception as e:
        print(f"✗ Error converting {html_path.name}: {e}")
        return False

def convert_with_playwright(html_path, pdf_path):
    """Convert HTML to PDF using playwright."""
    try:
        from playwright.sync_api import sync_playwright
        print(f"Converting {html_path.name} to PDF...")
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"file://{html_path.absolute()}")
            page.pdf(path=str(pdf_path), format="Letter", print_background=True)
            browser.close()
        return True
    except Exception as e:
        print(f"✗ Error converting {html_path.name}: {e}")
        return False

def convert_with_wkhtmltopdf(html_path, pdf_path):
    """Convert HTML to PDF using wkhtmltopdf."""
    try:
        print(f"Converting {html_path.name} to PDF...")
        subprocess.run([
            "wkhtmltopdf",
            "--enable-local-file-access",
            "--print-media-type",
            str(html_path),
            str(pdf_path)
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error converting {html_path.name}: {e}")
        return False

def merge_pdfs(pdf_files, output_path):
    """Merge multiple PDFs into one using PyPDF2."""
    try:
        import PyPDF2
    except ImportError:
        print("Installing PyPDF2 for PDF merging...")
        subprocess.run([sys.executable, "-m", "pip", "install", "PyPDF2"], 
                      check=True)
        import PyPDF2
    
    print(f"\nMerging {len(pdf_files)} PDFs into combined document...")
    merger = PyPDF2.PdfMerger()
    
    for pdf in sorted(pdf_files):
        print(f"  Adding: {pdf.name}")
        merger.append(str(pdf))
    
    merger.write(str(output_path))
    merger.close()
    print(f"✓ Combined PDF created: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert HTML notebooks to PDFs")
    parser.add_argument("--merge", action="store_true", 
                       help="Merge all PDFs into a single file")
    parser.add_argument("--tool", choices=["weasyprint", "wkhtmltopdf", "playwright", "auto"],
                       default="auto", help="PDF conversion tool to use")
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    html_dir = base_dir / "examples" / "published" / "html"
    pdf_dir = base_dir / "validation" / "python_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Check dependencies
    print("Checking PDF conversion tools...")
    available = check_dependencies()
    
    # Determine which tool to use
    if args.tool == "auto":
        if available['playwright']:
            tool = "playwright"
        elif available['weasyprint']:
            tool = "weasyprint"
        elif available['wkhtmltopdf']:
            tool = "wkhtmltopdf"
        else:
            print("No PDF conversion tool found. Playwright is already installed, using it.")
            tool = "playwright"
    else:
        tool = args.tool
        if not available[tool]:
            if tool == "weasyprint":
                if not install_weasyprint():
                    return 1
            else:
                print(f"✗ {tool} is not installed")
                return 1
    
    print(f"\nUsing {tool} for PDF conversion\n")
    
    # Convert HTML files to PDF
    html_files = list(html_dir.glob("*.html"))
    if not html_files:
        print(f"✗ No HTML files found in {html_dir}")
        return 1
    
    print(f"Found {len(html_files)} HTML files to convert\n")
    
    converted_pdfs = []
    for html_file in sorted(html_files):
        pdf_file = pdf_dir / html_file.with_suffix('.pdf').name
        
        if tool == "weasyprint":
            success = convert_with_weasyprint(html_file, pdf_file)
        elif tool == "playwright":
            success = convert_with_playwright(html_file, pdf_file)
        else:
            success = convert_with_wkhtmltopdf(html_file, pdf_file)
        
        if success:
            converted_pdfs.append(pdf_file)
            print(f"✓ Created: {pdf_file.name}\n")
    
    print(f"\nSuccessfully converted {len(converted_pdfs)}/{len(html_files)} files")
    
    # Optionally merge PDFs
    if args.merge and converted_pdfs:
        combined_pdf = base_dir / "validation" / "combined_python_examples.pdf"
        merge_pdfs(converted_pdfs, combined_pdf)
    
    # Copy MATLAB reference PDF
    matlab_pdf_src = base_dir.parent / "shmtools-matlab" / "SHMTools" / "Documentation" / "ExampleUsages.pdf"
    if matlab_pdf_src.exists():
        matlab_pdf_dst = base_dir / "validation" / "matlab_reference" / "ExampleUsages.pdf"
        matlab_pdf_dst.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(matlab_pdf_src, matlab_pdf_dst)
        print(f"\n✓ Copied MATLAB reference: {matlab_pdf_dst}")
    
    print("\n✓ PDF conversion complete!")
    print(f"\nPDFs saved to: {pdf_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())