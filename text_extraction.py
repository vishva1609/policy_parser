"""
Simple document processing with PyMuPDF.
"""
from src.pipeline import DocumentPipeline
from pathlib import Path
import sys


def main():
    """Run the pipeline."""
    
    # Initialize pipeline
    pipeline = DocumentPipeline(
        chunk_by="section",
        embedding_model="all-MiniLM-L6-v2",
        output_dir="./output"
    )
    
    # Check for PDF path argument
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if not Path(pdf_path).exists():
            print(f"Error: File not found: {pdf_path}")
            return
        
        # Process single PDF
        results = pipeline.process_document(pdf_path)
        
    else:
        # Process all PDFs in samples folder
        samples_dir = Path("./samples")
        pdf_files = list(samples_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("\nNo PDF files found in ./samples directory")
            print("Usage: python text_extraction.py <path_to_pdf>")
            print("   or: Place PDF files in ./samples directory")
            return
        
        print(f"\nFound {len(pdf_files)} PDF file(s)")
        for pdf_file in pdf_files:
            try:
                pipeline.process_document(str(pdf_file))
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
                continue
        
        # Show statistics
        print("\n" + "=" * 60)
        print("Pipeline Statistics")
        print("=" * 60)
        stats = pipeline.get_stats()
        print(f"Vector Database: {stats['vector_db']['total_chunks']} chunks")
        print(f"Knowledge Graph: {stats['knowledge_graph']['nodes']} nodes")
        print("=" * 60)
        
        # Example search
        if stats['vector_db']['total_chunks'] > 0:
            print("\n" + "=" * 60)
            print("Example Search")
            print("=" * 60)
            query = input("\nEnter search query (or press Enter to skip): ").strip()
            
            if query:
                results = pipeline.search(query, n_results=3)
                
                print(f"\nTop results for: '{query}'")
                print("-" * 60)
                
                for result in results['results']:
                    print(f"\n[{result['rank']}] Score: {result['score']}")
                    print(f"Document: {result['document']}")
                    print(f"Section: {result['section']}")
                    print(f"Pages: {result['pages']}")
                    print(f"Text: {result['text'][:200]}...")


if __name__ == "__main__":
    main()
