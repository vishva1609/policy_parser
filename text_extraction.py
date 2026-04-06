"""
Unified Document Processing Pipeline
- Basic: PDF parsing, chunking, embeddings, Excel search
- Advanced Mode: + Policy analysis and compliance questions (requires --advanced flag)

Hyperparameter Options:
  --interactive (-i)    : Prompt for hyperparameters at runtime
  --temperature=X       : Set randomness (0.0=focused, 1.0=creative, default 0.3)
  --max-tokens=N        : Max response length in tokens (default 2048)
  --top-p=X             : Nucleus sampling threshold (0.0-1.0, default 1.0)
    --frequency-penalty=X : Penalize repetition (-2.0 to 2.0, default 0.0)
    --presence-penalty=X  : Penalize already-present topics (-2.0 to 2.0, default 0.0)
    --seed=N              : Optional random seed for reproducible output
    --top-k=N             : Optional top-k sampling cutoff
  --model=NAME          : Mistral model name
  --auto-tune           : Automatically tune hyperparameters for best accuracy
  --compare-models      : Compare different models for efficiency

Architecture note:
  This script is the views/CLI entry point.
  It delegates to app.views.pipeline_runner.PipelineRunner (views layer),
  which calls app.service.document_service.DocumentService (service layer),
  which uses app.models.* (models layer).
  This mirrors the layered architecture of archit1012/qa-bot-llm.
"""
# ── New layered architecture imports ────────────────────────────────────────
from app.views.pipeline_runner import PipelineRunner
from app.service.document_service import DocumentService

from pathlib import Path
import sys
import json
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-rtv not required for standard mode

# Advanced components (optional) — now in app/models/
try:
    from app.models.policy_analyzer import PolicyAnalyzer
    from app.models.question_generator import (
        QuestionGenerator,
        AutoTuner,
        AdaptiveTuner,
        ModelComparator,
        list_available_models,
    )
    from app.common.excel_utils import export_compliance_questions_to_excel
    ADVANCED_AVAILABLE = True
except ImportError as e:
    ADVANCED_AVAILABLE = False
    if "--debug" in sys.argv:
        print(f"DEBUG: Advanced import error: {e}")


# Available Mistral models for selection
AVAILABLE_MODELS = [
    "mistral-small-latest",      # Fast, cost-effective
    "mistral-medium-latest",     # Balanced
    "mistral-large-latest",      # Highest quality
    "open-mistral-7b",           # Open source, fast
    "open-mixtral-8x7b",         # Open source, balanced
    "open-mixtral-8x22b",        # Open source, high quality
]


def get_model_tuning_config_path(output_dir: str, model_name: str) -> str:
    """Return model-specific tuning config path under output/tuning."""
    return str(AutoTuner.get_tuning_output_path(output_dir, model_name))


def get_model_output_suffix(model_name: str) -> str:
    """Return a filesystem-safe suffix for model-specific output files."""
    return AutoTuner._sanitize_model_name(model_name)


def get_interactive_hyperparameters() -> dict:
    """
    Prompt user for hyperparameters interactively at runtime.
    
    Returns:
        Dictionary with temperature, max_tokens, top_p, and model
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER CONFIGURATION")
    print("="*60)
    print("Press Enter to use default values.\n")
    
    # Temperature
    print("TEMPERATURE: Controls randomness (0.0=focused, 1.0=creative)")
    print("  - Lower (0.1-0.3): More consistent, focused compliance questions")
    print("  - Higher (0.5-0.8): More creative, varied questions")
    temp_input = input("Enter temperature [default: 0.3]: ").strip()
    temperature = float(temp_input) if temp_input else 0.3
    if not 0.0 <= temperature <= 1.0:
        print(f"  WARNING: Temperature {temperature} outside 0-1 range, clamping.")
        temperature = max(0.0, min(1.0, temperature))
    
    # Max tokens
    print("\nMAX TOKENS: Maximum response length (~4 chars per token)")
    print("  - 1024: Shorter responses")
    print("  - 2048: Standard (default)")
    print("  - 4096: Longer, more detailed responses")
    tokens_input = input("Enter max tokens [default: 2048]: ").strip()
    max_tokens = int(tokens_input) if tokens_input else 2048
    
    # Top P (nucleus sampling)
    print("\nTOP P: Nucleus sampling threshold")
    print("  - 1.0: Consider all tokens (default)")
    print("  - 0.9: Focus on top 90% probability mass")
    print("  - 0.5: Very focused, less diverse")
    print("  TIP: Don't change both temperature and top_p together.")
    top_p_input = input("Enter top_p [default: 1.0]: ").strip()
    top_p = float(top_p_input) if top_p_input else 1.0
    if not 0.0 <= top_p <= 1.0:
        print(f"  WARNING: top_p {top_p} outside 0-1 range, clamping.")
        top_p = max(0.0, min(1.0, top_p))

    # Frequency penalty
    print("\nFREQUENCY PENALTY: Penalizes repeated tokens")
    print("  - Range: -2.0 to 2.0")
    print("  - 0.0: No penalty (default)")
    freq_input = input("Enter frequency_penalty [default: 0.0]: ").strip()
    frequency_penalty = float(freq_input) if freq_input else 0.0

    # Presence penalty
    print("\nPRESENCE PENALTY: Penalizes topics already used")
    print("  - Range: -2.0 to 2.0")
    print("  - 0.0: No penalty (default)")
    pres_input = input("Enter presence_penalty [default: 0.0]: ").strip()
    presence_penalty = float(pres_input) if pres_input else 0.0

    # Seed
    print("\nSEED: Optional deterministic generation seed")
    print("  - Leave blank for random behavior")
    seed_input = input("Enter seed [default: blank]: ").strip()
    seed = int(seed_input) if seed_input else None

    # Top K
    print("\nTOP K: Optional top-k token cutoff")
    print("  - Leave blank to disable")
    top_k_input = input("Enter top_k [default: blank]: ").strip()
    top_k = int(top_k_input) if top_k_input else None
    
    # Model selection
    print("\nMODEL SELECTION:")
    for i, model_name in enumerate(AVAILABLE_MODELS, 1):
        quality = "Fast/Cheap" if "small" in model_name or "7b" in model_name else \
                  "High Quality" if "large" in model_name or "22b" in model_name else "Balanced"
        print(f"  {i}. {model_name} ({quality})")
    
    model_input = input("Enter model number or name [default: 1]: ").strip()
    if model_input.isdigit() and 1 <= int(model_input) <= len(AVAILABLE_MODELS):
        model = AVAILABLE_MODELS[int(model_input) - 1]
    elif model_input in AVAILABLE_MODELS:
        model = model_input
    elif model_input:
        model = model_input  # Allow custom model names
    else:
        model = AVAILABLE_MODELS[0]
    
    print("\n" + "-"*60)
    print("SELECTED CONFIGURATION:")
    print(f"  Temperature: {temperature}")
    print(f"  Max Tokens:  {max_tokens}")
    print(f"  Top P:       {top_p}")
    print(f"  Freq Penalty:{frequency_penalty}")
    print(f"  Pres Penalty:{presence_penalty}")
    print(f"  Seed:        {seed}")
    print(f"  Top K:       {top_k}")
    print(f"  Model:       {model}")
    print("-"*60)
    
    confirm = input("Proceed with these settings? [Y/n]: ").strip().lower()
    if confirm == 'n':
        return get_interactive_hyperparameters()  # Recurse to re-enter
    
    return {
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'frequency_penalty': frequency_penalty,
        'presence_penalty': presence_penalty,
        'seed': seed,
        'top_k': top_k,
        'model': model
    }


def process_advanced(
    chunks,
    output_dir,
    temperature=0.3,
    max_tokens=2048,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    seed=None,
    top_k=None,
    model="mistral-small-latest"
):
    """
    Process chunks to generate compliance questions

    Args:
        chunks: List of document chunks
        output_dir: Output directory path
        temperature: LLM temperature (0.0-1.0). Lower = more focused.
        max_tokens: Max response length in tokens.
        top_p: Nucleus sampling threshold (0.0-1.0).
        frequency_penalty: Penalize repetition.
        presence_penalty: Penalize repeating existing topics.
        seed: Optional random seed.
        top_k: Optional top-k sampling cutoff.
        model: Mistral model name.

    Returns:
        Dictionary with questions, statements, and output file paths, or None if failed
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
        generator = QuestionGenerator(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            top_k=top_k,
            model=model
        )
        
        questions = generator.generate_questions_from_statements(
            requirements,
            questions_per_statement=3,
            batch_size=10
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
        model_suffix = get_model_output_suffix(model)
        
        # Export questions to JSON
        questions_file = output_path / f"compliance_questions_{model_suffix}.json"
        with open(questions_file, 'w', encoding='utf-8') as f:
            json.dump(generator.export_questions(questions), f, indent=2)
        print(f"   Saved questions to: {questions_file}")
        
        # Export policy statements
        statements_file = output_path / f"policy_statements_{model_suffix}.json"
        with open(statements_file, 'w', encoding='utf-8') as f:
            json.dump(analyzer.export_statements(statements), f, indent=2)
        print(f"   Saved statements to: {statements_file}")
        
        # Export to Excel
        excel_file = output_path / f"compliance_questions_{model_suffix}.xlsx"
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
            'requirements': requirements,
            'questions_file': str(questions_file),
            'statements_file': str(statements_file),
            'excel_file': str(excel_file),
            'model': model,
        }
        
    except Exception as e:
        print(f"ERROR: Processing failed: {e}")
        return None





def main():
    # Parse command line arguments
    advanced_mode = '--advanced' in sys.argv or '--ai' in sys.argv or '-a' in sys.argv
    interactive_mode = '--interactive' in sys.argv or '-i' in sys.argv
    auto_tune_mode = '--auto-tune' in sys.argv or '--autotune' in sys.argv
    compare_models_mode = '--compare-models' in sys.argv or '--compare' in sys.argv
    list_models_mode = '--list-models' in sys.argv

    # Handle --list-models flag
    if list_models_mode and ADVANCED_AVAILABLE:
        list_available_models()
        return

    # Default hyperparameters
    temperature = 0.3
    max_tokens = 2048
    top_p = 1.0
    frequency_penalty = 0.0
    presence_penalty = 0.0
    seed = None
    top_k = None
    model = "mistral-small-latest"
    
    # Interactive mode: prompt for hyperparameters at runtime
    if interactive_mode and advanced_mode:
        params = get_interactive_hyperparameters()
        temperature = params['temperature']
        max_tokens = params['max_tokens']
        top_p = params['top_p']
        frequency_penalty = params['frequency_penalty']
        presence_penalty = params['presence_penalty']
        seed = params['seed']
        top_k = params['top_k']
        model = params['model']
        positional_args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    else:
        # Parse from command line arguments (supports both --arg=value and --arg value)
        positional_args = []
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            arg = args[i]

            if arg.startswith('--temperature='):
                temperature = float(arg.split('=', 1)[1])
            elif arg == '--temperature' and i + 1 < len(args):
                i += 1
                temperature = float(args[i])
            elif arg.startswith('--max-tokens='):
                max_tokens = int(arg.split('=', 1)[1])
            elif arg == '--max-tokens' and i + 1 < len(args):
                i += 1
                max_tokens = int(args[i])
            elif arg.startswith('--top-p='):
                top_p = float(arg.split('=', 1)[1])
            elif arg == '--top-p' and i + 1 < len(args):
                i += 1
                top_p = float(args[i])
            elif arg.startswith('--frequency-penalty='):
                frequency_penalty = float(arg.split('=', 1)[1])
            elif arg == '--frequency-penalty' and i + 1 < len(args):
                i += 1
                frequency_penalty = float(args[i])
            elif arg.startswith('--presence-penalty='):
                presence_penalty = float(arg.split('=', 1)[1])
            elif arg == '--presence-penalty' and i + 1 < len(args):
                i += 1
                presence_penalty = float(args[i])
            elif arg.startswith('--seed='):
                seed = int(arg.split('=', 1)[1])
            elif arg == '--seed' and i + 1 < len(args):
                i += 1
                seed = int(args[i])
            elif arg.startswith('--top-k='):
                top_k = int(arg.split('=', 1)[1])
            elif arg == '--top-k' and i + 1 < len(args):
                i += 1
                top_k = int(args[i])
            elif arg.startswith('--model='):
                model = arg.split('=', 1)[1]
            elif arg == '--model' and i + 1 < len(args):
                i += 1
                model = args[i]
            elif not arg.startswith('-'):
                positional_args.append(arg)

            i += 1

    if advanced_mode and not ADVANCED_AVAILABLE:
        print("ERROR: Advanced mode requires additional packages.")
        return

    # Get PDF path from positional arguments
    if positional_args:
        pdf_path = positional_args[0]
    else:
        pdf_path = "samples/Information Security & Management Policy v3.pdf"
    
    if not Path(pdf_path).exists():
        print(f"ERROR: File not found: {pdf_path}")
        print("\nUsage: python text_extraction.py [pdf_path] [--advanced] [options]")
        print("\nModes:")
        print("  --advanced, -a        : Enable policy analysis and compliance question generation")
        print("  --interactive, -i     : Prompt for hyperparameters interactively at runtime")
        print("  --auto-tune           : Automatically find optimal hyperparameters")
        print("  --compare-models      : Compare different models for efficiency")
        print("  --list-models         : List all available models")
        print("\nHyperparameters:")
        print("  --temperature=X       : Set randomness (0.0=focused, 1.0=creative, default 0.3)")
        print("  --max-tokens=N        : Max response length in tokens (default 2048)")
        print("  --top-p=X             : Nucleus sampling threshold (0.0-1.0, default 1.0)")
        print("  --frequency-penalty=X : Penalize repetition (-2.0 to 2.0, default 0.0)")
        print("  --presence-penalty=X  : Penalize already-present topics (-2.0 to 2.0, default 0.0)")
        print("  --seed=N              : Optional seed for reproducibility")
        print("  --top-k=N             : Optional top-k sampling cutoff")
        print("  --model=NAME          : Mistral model name")
        print("\nAvailable Models:")
        for m in AVAILABLE_MODELS:
            print(f"  - {m}")
        print("\nExamples:")
        print("  python text_extraction.py doc.pdf --advanced -i")
        print("  python text_extraction.py doc.pdf --advanced --auto-tune")
        print("  python text_extraction.py doc.pdf --advanced --compare-models")
        print("  python text_extraction.py doc.pdf --advanced --model=mistral-large-latest")
        return
    
    # Header
    print("\n" + "="*80)
    if advanced_mode:
        print("ADVANCED POLICY DOCUMENT PROCESSING")
    else:
        print("DOCUMENT PROCESSING PIPELINE")
    print("="*80)
    print(f"Mode: {'Advanced + Compliance Analysis' if advanced_mode else 'Standard Processing'}")
    if advanced_mode:
        print(
            f"Temperature: {temperature} | Max Tokens: {max_tokens} | Top P: {top_p} | "
            f"Freq Penalty: {frequency_penalty} | Presence Penalty: {presence_penalty} | "
            f"Seed: {seed} | Top K: {top_k}"
        )
        print(f"Model: {model}")
    print(f"File: {pdf_path}")
    print("="*80)
    
    # Single output directory
    output_dir = "output"

    # Initialize pipeline using DocumentService (app/service layer)
    service = DocumentService(
        output_dir=output_dir,
        max_chunk_chars=3000,
        min_chunk_chars=500
    )

    try:
        # Process with intelligent chunking
        results = service.process(
            pdf_path,
            output_format="txt",
            overwrite_files=False
        )

        # Display statistics
        print("\nPROCESSING STATISTICS:")
        print("-" * 80)
        print(f"  Total Pages:        {results['stats']['total_pages']}")
        print(f"  Total Sections:     {results['stats']['total_sections']}")
        print(f"  Total Chunks:       {results['stats']['total_chunks']}")
        print(f"  Avg Chunk Size:     {results['stats']['avg_chunk_size']} chars")
        print(f"  Chunk Files:        {len(results.get('chunk_files', []))}")
        print("-" * 80)
        
        # Advanced Processing (if enabled)
        if advanced_mode:
            print("\n" + "="*80)
            print("POLICY ANALYSIS")
            print("="*80)
            
            # Auto-tune mode: find optimal hyperparameters
            if auto_tune_mode:
                print("\nAUTO-TUNING: Evaluating available models...")
                print("This will choose the best model and hyperparameters based on tuning score.\n")

                best_model = None
                best_params = None
                best_score = -1.0
                model_results = []
                failed_models = []

                for candidate_model in AVAILABLE_MODELS:
                    print(f"\n{'-' * 80}")
                    print(f"Testing model: {candidate_model}")
                    print(f"{'-' * 80}")

                    try:
                        tuning_config_path = AutoTuner.get_tuning_output_path(output_dir, candidate_model)
                        adaptive = AdaptiveTuner(config_path=tuning_config_path)

                        if adaptive.custom_config:
                            # Cached tuning found - reuse it
                            params = adaptive.get_optimal_params(chunks=results['chunks'])
                            candidate_score = adaptive.custom_config.get('best_params', {}).get('score', 0.0)
                            print(f"✓ Reusing cached tuning results from: {tuning_config_path.name}")
                            print(f"  Cached score: {candidate_score:.2f}")
                        else:
                            # No cached results - run fresh tuning
                            print(f"ℹ No cached tuning found for '{candidate_model}'")
                            print(f"  Running fresh tuning (this may take several minutes)...")
                            
                            tuner = AutoTuner(
                                model=candidate_model,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                seed=seed,
                                top_k=top_k,
                            )
                            
                            # Use better sample size: 20% of chunks, minimum 15
                            sample_size = max(15, len(results['chunks']) // 5)
                            print(f"  Using {sample_size} chunks for tuning (out of {len(results['chunks'])} total)")
                            
                            params, trial_results = tuner.tune(
                                results['chunks'][:sample_size],
                                output_dir=output_dir,
                                max_trials=9
                            )
                            
                            # Validate tuning results
                            if params is None or params.get('score', 0) <= 0:
                                print(f"⚠ Tuning produced invalid results, using defaults for {candidate_model}")
                                params = {
                                    'temperature': 0.3,
                                    'max_tokens': 2048,
                                    'top_p': 1.0,
                                    'frequency_penalty': 0.0,
                                    'presence_penalty': 0.0,
                                    'seed': None,
                                    'top_k': None,
                                    'score': 0.0
                                }
                                candidate_score = 0.0
                            else:
                                candidate_score = params['score']

                        model_results.append({
                            'model': candidate_model,
                            'score': candidate_score,
                            'params': params,
                        })

                        print(f"✓ Best score for {candidate_model}: {candidate_score:.2f}")

                        if candidate_score > best_score:
                            best_score = candidate_score
                            best_model = candidate_model
                            best_params = params

                    except Exception as e:
                        error_msg = str(e)[:100]
                        print(f"✗ Error testing model {candidate_model}: {error_msg}")
                        failed_models.append((candidate_model, error_msg))
                        # Continue to next model instead of failing
                        continue

                # Validate we have at least one successful model
                if best_model is None or best_params is None:
                    if failed_models:
                        print(f"\n❌ ERROR: All models failed to tune:")
                        for model, error in failed_models:
                            print(f"   - {model}: {error}")
                    raise RuntimeError("Auto-tuning failed to select a valid model. Check API key and quota.")

                model = best_model
                temperature = best_params.get('temperature', 0.3)
                max_tokens = best_params.get('max_tokens', 2048)
                top_p = best_params.get('top_p', 1.0)
                frequency_penalty = best_params.get('frequency_penalty', 0.0)
                presence_penalty = best_params.get('presence_penalty', 0.0)
                seed = best_params.get('seed', None)
                top_k = best_params.get('top_k', None)

                print(f"\n{'=' * 80}")
                print("BEST AUTO-TUNED MODEL SELECTION")
                print(f"{'=' * 80}")
                for item in model_results:
                    marker = "★ " if item['model'] == best_model else "  "
                    print(f"{marker}{item['model']}: {item['score']:.2f}")
                print(f"\nSelected model: {best_model} ★")
                print(f"Selected score: {best_score:.2f}")
                print("Optimal parameters:")
                print(f"  Temperature:     {temperature}")
                print(f"  Max Tokens:      {max_tokens}")
                print(f"  Top P:           {top_p}")
                print(f"  Freq Penalty:    {frequency_penalty}")
                print(f"  Pres Penalty:    {presence_penalty}")
                print(f"  Seed:            {seed}")
                print(f"  Top K:           {top_k}")
                print("=" * 80)
            
            # Model comparison mode
            if compare_models_mode:
                print("\nMODEL COMPARISON: Testing different models...")
                print("This will compare efficiency and quality across models.\n")
                
                comparator = ModelComparator(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    seed=seed,
                    top_k=top_k,
                )
                comparison_report = comparator.compare(
                    results['chunks'],
                    questions_per_statement=2,
                    max_statements=5,
                    output_path=f"{output_dir}/model_comparison.json"
                )
                
                print(f"\nBest model for this document: {comparison_report.best_model}")
                
                # Ask if user wants to use the best model
                use_best = input(f"\nUse {comparison_report.best_model} for full generation? [Y/n]: ").strip().lower()
                if use_best != 'n':
                    model = comparison_report.best_model
            
            advanced_results = process_advanced(
                results['chunks'],
                output_dir,
                temperature,
                max_tokens,
                top_p,
                frequency_penalty,
                presence_penalty,
                seed,
                top_k,
                model,
            )
            
            if advanced_results:
                print("\nAnalysis complete!")
                results['compliance_questions'] = advanced_results['questions']
                results['policy_statements'] = advanced_results['statements']
                results['questions_file'] = advanced_results['questions_file']
                results['statements_file'] = advanced_results['statements_file']
                results['advanced_excel_file'] = advanced_results['excel_file']
                results['advanced_model'] = advanced_results['model']
        
        # Summary
        print("\nOUTPUT FILES:")
        print("-" * 80)
        print(f"  Excel Index  : {results.get('excel_path', 'N/A')}")
        print(f"  Chunk Files  : {len(results.get('chunk_files', []))} files in {output_dir}/chunks/")
        print(f"  Vector DB    : {output_dir}/vectordb/")
        print(f"  Parsed JSON  : {output_dir}/[document]_parsed.json")
        
        if advanced_mode and 'compliance_questions' in results:
            print(f"\nADVANCED OUTPUTS:")
            print(f"Model Used:      {results.get('advanced_model', model)}")
            print(f"Questions:       {results.get('questions_file', output_dir + '/compliance_questions.json')}")
            print(f"Excel Report:    {results.get('advanced_excel_file', output_dir + '/compliance_questions.xlsx')}")
            print(f"Statements:      {results.get('statements_file', output_dir + '/policy_statements.json')}")
        
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
