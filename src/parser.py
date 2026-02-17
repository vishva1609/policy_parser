"""
Simple PDF parser using PyMuPDF.
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List
import uuid
import re

from .schemas import DocumentElement, DocumentSection, ParsedDocument


class SimplePDFParser:
    """Parse PDFs using PyMuPDF - fast and reliable."""
    
    def __init__(self):
        """Initialize parser."""
        self.heading_patterns = [
            r'^#+\s+',  # Markdown-style headings
            r'^\d+\.\s+[A-Z]',  # Numbered sections
            r'^[A-Z][A-Z\s]{3,}$',  # ALL CAPS headings
        ]
    
    def parse_pdf(self, pdf_path: str) -> ParsedDocument:
        """
        Parse a PDF file into structured format.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ParsedDocument with sections and elements
        """
        pdf_path = Path(pdf_path)
        document_id = str(uuid.uuid4())
        
        print(f"\nParsing PDF: {pdf_path.name}")
        print("-" * 60)
        
        # Open PDF
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        print(f"Pages: {total_pages}")
        
        # Extract text from all pages
        all_sections = []
        current_section = None
        section_counter = 0
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            # Get text blocks with formatting info
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        # Extract text from spans
                        text_parts = []
                        is_bold = False
                        font_size = 12
                        
                        for span in line.get("spans", []):
                            text_parts.append(span.get("text", ""))
                            font_size = max(font_size, span.get("size", 12))
                            if "bold" in span.get("font", "").lower():
                                is_bold = True
                        
                        text = " ".join(text_parts).strip()
                        
                        if not text:
                            continue
                        
                        # Determine if this is a heading
                        is_heading = (
                            is_bold and font_size > 13 or
                            self._looks_like_heading(text)
                        )
                        
                        if is_heading:
                            # Start new section
                            if current_section and current_section.elements:
                                all_sections.append(current_section)
                            
                            section_counter += 1
                            current_section = DocumentSection(
                                title=text,
                                level=1,
                                elements=[],
                                page_start=page_num + 1,
                                page_end=page_num + 1
                            )
                        else:
                            # Add to current section
                            if current_section is None:
                                section_counter += 1
                                current_section = DocumentSection(
                                    title="Document Content",
                                    level=1,
                                    elements=[],
                                    page_start=page_num + 1,
                                    page_end=page_num + 1
                                )
                            
                            element = DocumentElement(
                                text=text,
                                element_type="paragraph",
                                page=page_num + 1,
                                metadata={"font_size": font_size}
                            )
                            current_section.elements.append(element)
                            current_section.page_end = page_num + 1
        
        # Add last section
        if current_section and current_section.elements:
            all_sections.append(current_section)
        
        doc.close()
        
        print(f"Extracted {len(all_sections)} sections")
        print(f"Parsing complete\n")
        
        return ParsedDocument(
            document_id=document_id,
            filename=pdf_path.name,
            total_pages=total_pages,
            sections=all_sections,
            metadata={"parser": "pymupdf"}
        )
    
    def _looks_like_heading(self, text: str) -> bool:
        """Check if text looks like a heading."""
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                return True
        return False
