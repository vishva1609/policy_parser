"""
Unified Document Processing Pipeline
- Basic: PDF parsing, chunking, embeddings, Excel search
- Advanced Mode: + Policy analysis and compliance questions (requires --advanced flag)
"""
from src.pipeline import DocumentPipeline
from pathlib import Path
import sys
import json
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not required for standard mode

# Advanced components (optional)
try:
    from src.policy_analyzer import PolicyAnalyzer
    from src.question_generator import QuestionGenerator
    from src.excel_generator import export_compliance_questions_to_excel
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False


def process_advanced(chunks, output_dir):
    """
    Process chunks to generate compliance questions
    
    Args:
        chunks: List of document chunks
        output_dir: Output directory path
        
    Returns:
        Dictionary with questions and statements, or None if failed
    """
    try:
        # Step 1: Analyze policy statements
        print("\n[1/3] Analyzing policy statements...")
        analyzer = PolicyAnalyzer()
        statements = analyzer.analyze_document(chunks)
        
        print(f"   Found {len(statements)} policy statements")
        
        # Get requirements
        requirements = analyzer.get_requirements(statements, min_confidence=0.5)
        print(f"   Identified {len(requirements)} requirement statements")
        
        # Display categories
        categorized = analyzer.get_statements_by_category(requirements)
        print(f"\n   Requirement statements by category:")
        for category, stmts in categorized.items():
            print(f"     - {category.value}: {len(stmts)} statements")
        
        # Step 2: Generate compliance questions
        print(f"\n[2/3] Generating compliance questions...")
        generator = QuestionGenerator()
        
        questions = generator.generate_questions_from_statements(
            requirements,
            questions_per_statement=3,
            batch_size=5
        )
        
        if not questions:
            print("WARNING: No questions generated.")
            print("\nThis usually means your API quota is exhausted.")
            print("Standard document processing completed successfully.")
            print("Advanced features will be available once your quota resets.")
            return None
        
        print(f"\nGenerated {len(questions)} compliance questions")
        
        # Group by category
        questions_by_category = {}
        for q in questions:
            if q.category not in questions_by_category:
                questions_by_category[q.category] = []
            questions_by_category[q.category].append(q)
        
        print(f"\n   Questions by category:")
        for category, qs in sorted(questions_by_category.items()):
            print(f"     - {category}: {len(qs)} questions")
        
        # Step 3: Export results
        print(f"\n[3/3] Exporting results...")
        output_path = Path(output_dir)
        
        # Export questions to JSON
        questions_file = output_path / "compliance_questions.json"
        with open(questions_file, 'w', encoding='utf-8') as f:
            json.dump(generator.export_questions(questions), f, indent=2)
        print(f"   Saved questions to: {questions_file}")
        
        # Export policy statements
        statements_file = output_path / "policy_statements.json"
        with open(statements_file, 'w', encoding='utf-8') as f:
            json.dump(analyzer.export_statements(statements), f, indent=2)
        print(f"   Saved statements to: {statements_file}")
        
        # Export to Excel
        excel_file = output_path / "compliance_questions.xlsx"
        export_compliance_questions_to_excel(questions, requirements, excel_file)
        print(f"   Saved Excel report to: {excel_file}")
        
        # Display sample questions
        print(f"\n   Sample questions (first 3):")
        for i, q in enumerate(questions[:3], 1):
            print(f"\n   {i}. {q.question}")
            print(f"      Category: {q.category} | Type: {q.question_type}")
        
        return {
            'questions': questions,
            'statements': statements,
            'requirements': requirements
        }
        
    except Exception as e:
        print(f"ERROR: Processing failed: {e}")
        return None





def main():
    # Parse command line arguments
    advanced_mode = '--advanced' in sys.argv or '--ai' in sys.argv or '-a' in sys.argv
    
    if advanced_mode and not ADVANCED_AVAILABLE:
        print("ERROR: Advanced mode requires additional packages.")
        return
    
    # Get PDF path
    pdf_args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    if pdf_args:
        pdf_path = pdf_args[0]
    else:
        pdf_path = "samples/Information Security & Management Policy v3.pdf"
    
    if not Path(pdf_path).exists():
        print(f"ERROR: File not found: {pdf_path}")
        print("\nUsage: python text_extraction.py [pdf_path] [--advanced]")
        print("  --advanced : Enable policy analysis and compliance question generation")
        return
    
    # Header
    print("\n" + "="*80)
    if advanced_mode:
        print("ADVANCED POLICY DOCUMENT PROCESSING")
    else:
        print("DOCUMENT PROCESSING PIPELINE")
    print("="*80)
    print(f"Mode: {'Advanced + Compliance Analysis' if advanced_mode else 'Standard Processing'}")
    print(f"File: {pdf_path}")
    print("="*80)
    
    # Single output directory
    output_dir = "output"
    
    # Initialize pipeline
    pipeline = DocumentPipeline(
        output_dir=output_dir,
        max_chunk_chars=3000,
        min_chunk_chars=500
    )
    
    try:
        # Process with intelligent chunking
        results = pipeline.process_document(
            pdf_path,
            output_format="txt",      # or "json" or "md"
            overwrite_files=False     # Don't overwrite existing files
        )
        
        # Display statistics
        print("\nPROCESSING STATISTICS:")
        print("-" * 80)
        print(f"  Total Pages:        {results['stats']['total_pages']}")
        print(f"  Total Sections:     {results['stats']['total_sections']}")
        print(f"  Total Chunks:       {results['stats']['total_chunks']}")
        print(f"  Avg Chunk Size:     {results['stats']['avg_chunk_size']} chars")
        print(f"  Files Created:      {results['stats']['total_files_created']}")
        print("-" * 80)
        
        # Advanced Processing (if enabled)
        if advanced_mode:
            print("\n" + "="*80)
            print("POLICY ANALYSIS")
            print("="*80)
            
            advanced_results = process_advanced(results['chunks'], output_dir)
            
            if advanced_results:
                print("\nAnalysis complete!")
                results['compliance_questions'] = advanced_results['questions']
                results['policy_statements'] = advanced_results['statements']
        
        # Summary
        print("\nOUTPUT FILES:")
        print("-" * 80)
        print(f"Excel Index:     {results['excel_path']}")
        print(f"Chunk Files:     {len(results['chunk_files'])} files in {output_dir}/chunks/")
        print(f"Chunk Index:     {results['index_path']}")
        print(f"Vector DB:       {output_dir}/vectordb/")
        print(f"JSON Outputs:    {output_dir}/[document]_parsed.json, _graph.json")
        
        if advanced_mode and 'compliance_questions' in results:
            print(f"\nADVANCED OUTPUTS:")
            print(f"Questions:       {output_dir}/compliance_questions.json")
            print(f"Excel Report:    {output_dir}/compliance_questions.xlsx")
            print(f"Statements:      {output_dir}/policy_statements.json")
        
        print("-" * 80)
        
        print("\nProcessing complete!\n")
        print("TIP: Open the Excel file to search content and find page numbers quickly!")
        if not advanced_mode:
            print("TIP: Use --advanced flag for policy compliance questions")
        print("TIP: Existing chunk files were not overwritten. Use overwrite_files=True to replace.\n")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
