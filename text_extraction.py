"""
Enhanced Text Extraction with Intelligent Chunking and Excel Search
"""
from src.pipeline import DocumentPipeline
from pathlib import Path
import sys


def main():
    print("\n" + "="*80)
    print("ENHANCED DOCUMENT PROCESSING - INTELLIGENT CHUNKING")
    print("="*80)
    
    # Initialize pipeline
    pipeline = DocumentPipeline(
        output_dir="output",
        max_chunk_chars=3000,  # Maximum 3000 characters per chunk
        min_chunk_chars=500    # Minimum 500 characters
    )
    
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "samples/Information Security & Management Policy v3.pdf"
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        return
    
    try:
        # Process with intelligent chunking
        results = pipeline.process_document(
            pdf_path,
            output_format="txt",      # or "json" or "md"
            overwrite_files=False     # Don't overwrite existing files
        )
        
        # Display statistics
        print("\n📊 PROCESSING STATISTICS:")
        print("-" * 80)
        print(f"  Total Pages:        {results['stats']['total_pages']}")
        print(f"  Total Sections:     {results['stats']['total_sections']}")
        print(f"  Total Chunks:       {results['stats']['total_chunks']}")
        print(f"  Avg Chunk Size:     {results['stats']['avg_chunk_size']} chars")
        print(f"  Files Created:      {results['stats']['total_files_created']}")
        print("-" * 80)
        
        # Example: Search in Excel
        print("\n🔍 EXCEL SEARCH EXAMPLE:")
        print("-" * 80)
        search_term = "security"
        print(f"Searching for: '{search_term}'")
        excel_results = pipeline.search_in_excel(results['excel_path'], search_term)
        
        # Example: Semantic search
        print("\n🔍 SEMANTIC SEARCH EXAMPLE:")
        print("-" * 80)
        query = "information security policy"
        print(f"Query: '{query}'")
        search_results = pipeline.search(query, n_results=3)
        
        if search_results and 'results' in search_results:
            print(f"\nTop 3 semantic search results:\n")
            for result in search_results['results'][:3]:
                print(f"[{result['rank']}] Score: {result['score']}")
                print(f"    Section: {result.get('section_title', 'N/A')}")
                print(f"    Pages: {result.get('pages', 'N/A')}")
                print(f"    Text: {result['text'][:120]}...")
                print()
        
        # Summary
        print("\n📂 OUTPUT FILES:")
        print("-" * 80)
        print(f"✓ Excel Index:     {results['excel_path']}")
        print(f"✓ Chunk Files:     {len(results['chunk_files'])} files in output/chunks/")
        print(f"✓ Chunk Index:     {results['index_path']}")
        print(f"✓ Vector DB:       output/vectordb/")
        print(f"✓ JSON Outputs:    output/[document]_parsed.json, _chunks.json, _graph.json")
        print("-" * 80)
        
        print("\n✅ Processing complete!\n")
        print("💡 TIP: Open the Excel file to search content and find page numbers quickly!")
        print("💡 TIP: Existing chunk files were not overwritten. Use overwrite_files=True to replace.\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
