"""
Excel Generator for Searchable Document Index
"""
import pandas as pd
from pathlib import Path
from typing import List
import re
from .schemas import ParsedDocument, DocumentChunk


class ExcelGenerator:
    """Generate Excel file with searchable key-value pairs and page numbers"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def clean_text_for_excel(text: str) -> str:
        """Remove illegal characters for Excel cells"""
        # Remove control characters and non-printable characters
        # Keep only printable ASCII and common Unicode
        cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        # Replace problematic Unicode characters
        cleaned = cleaned.encode('ascii', 'ignore').decode('ascii')
        return cleaned
    
    def generate_searchable_index(
        self, 
        document: ParsedDocument,
        chunks: List[DocumentChunk],
        output_filename: str = None
    ) -> str:
        """
        Generate Excel file with searchable content and page numbers.
        
        Returns:
            Path to generated Excel file
        """
        if output_filename is None:
            base_name = Path(document.filename).stem
            output_filename = f"{base_name}_searchable_index.xlsx"
        
        excel_path = self.output_dir / output_filename
        
        # Prepare data for Excel
        rows = []
        
        # Add document-level metadata
        rows.append({
            'Type': 'Document',
            'Section': 'Metadata',
            'Content': f"Document: {document.filename}",
            'Page Numbers': f"1-{document.total_pages}",
            'Chunk ID': document.document_id,
            'Keywords': 'document, metadata'
        })
        
        # Add section-level information
        for section in document.sections:
            rows.append({
                'Type': 'Section',
                'Section': section.title,
                'Content': f"Section: {section.title}",
                'Page Numbers': f"{section.page_start}-{section.page_end}",
                'Chunk ID': '',
                'Keywords': section.title.lower()
            })
        
        # Add chunk-level information
        for i, chunk in enumerate(chunks, 1):
            # Clean text for Excel
            clean_text = self.clean_text_for_excel(chunk.text)
            
            # Extract keywords
            words = clean_text.split()
            keywords = ' '.join([w.lower() for w in words[:10] if len(w) > 3])
            
            # Get page range
            if chunk.pages:
                page_range = f"{min(chunk.pages)}-{max(chunk.pages)}" if len(chunk.pages) > 1 else str(chunk.pages[0])
            else:
                page_range = "N/A"
            
            # Get section name
            section_name = chunk.parent_context[0] if chunk.parent_context else 'N/A'
            section_name = self.clean_text_for_excel(section_name)
            
            rows.append({
                'Type': 'Chunk',
                'Section': section_name,
                'Content': clean_text[:500] + '...' if len(clean_text) > 500 else clean_text,
                'Page Numbers': page_range,
                'Chunk ID': chunk.chunk_id,
                'Keywords': keywords
            })
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Create Excel with multiple sheets
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main searchable index
            df.to_excel(writer, sheet_name='Searchable Index', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Pages', 'Total Sections', 'Total Chunks', 'Document Name'],
                'Value': [document.total_pages, len(document.sections), len(chunks), document.filename]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Page-to-content mapping
            page_mapping = []
            for chunk in chunks:
                if chunk.pages:
                    for page in chunk.pages:
                        section_name = chunk.parent_context[0] if chunk.parent_context else 'N/A'
                        section_name = self.clean_text_for_excel(section_name)
                        clean_preview = self.clean_text_for_excel(chunk.text[:200])
                        
                        page_mapping.append({
                            'Page Number': page,
                            'Section': section_name,
                            'Chunk ID': chunk.chunk_id,
                            'Content Preview': clean_preview + '...'
                        })
            
            if page_mapping:
                page_df = pd.DataFrame(page_mapping).sort_values('Page Number')
                page_df.to_excel(writer, sheet_name='Page Mapping', index=False)
        
        print(f"✓ Excel index generated: {excel_path}")
        return str(excel_path)
    
    def search_excel(self, excel_path: str, search_term: str) -> pd.DataFrame:
        """Search the Excel file for a term and return matching rows"""
        df = pd.read_excel(excel_path, sheet_name='Searchable Index')
        
        # Search in Content and Keywords columns
        mask = df['Content'].str.contains(search_term, case=False, na=False) | \
               df['Keywords'].str.contains(search_term, case=False, na=False)
        
        results = df[mask]
        return results
