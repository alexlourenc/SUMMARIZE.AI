import os
from dotenv import load_dotenv
from src.engine.ingestion import DataIngestion
from src.engine.processor import TextProcessor
from src.engine.summarizer import SummarizationEngine

# 5. MAIN PIPELINE EXECUTION
# 5. EXECUÇÃO DO PIPELINE PRINCIPAL

def run_summarize_ai(file_path: str, mode: str = "executive"):
    """
    Orchestrates the full flow: Ingestion -> Processing -> Summarization.
    Orquestra o fluxo completo: Ingestão -> Processamento -> Sumarização.
    """
    
    # Load environment variables from .env file
    # Carrega variáveis de ambiente do arquivo .env
    load_dotenv()
    
    # Initialize components following clean architecture
    # Inicializa componentes seguindo a arquitetura limpa
    ingestor = DataIngestion()
    processor = TextProcessor(chunk_size=2000, chunk_overlap=200)
    
    try:
        engine = SummarizationEngine()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return
    
    print(f"\n--- Starting SUMMARIZE.AI | Mode: {mode.upper()} ---")
    
    # 1. Ingestion: Detects type based on extension
    # 1. Ingestão: Detecta tipo com base na extensão
    source_type = "pdf" if file_path.endswith(".pdf") else "txt"
    docs = ingestor.load_data(file_path, source_type)
    
    if not docs:
        print("Process aborted: No data loaded. / Processo abortado: Sem dados carregados.")
        return

    # 2. Chunking: Split long text into manageable parts
    # 2. Chunking: Divide textos longos em partes gerenciáveis
    chunks = processor.process_documents(docs)
    
    # 3 & 4. Execution: Summarize using the engine (internalizes prompt logic)
    # 3 & 4. Execução: Sumariza usando o motor (internaliza a lógica de prompts)
    print("Executing LLM Chains... Please wait. / Executando Cadeias de LLM... Por favor, aguarde.")
    summary = engine.generate_summary(chunks, mode=mode)
    
    # 5. Output: Professional presentation
    # 5. Saída: Apresentação profissional
    print("\n" + "="*50)
    print("FINAL SUMMARY / RESUMO FINAL")
    print("="*50)
    print(summary)
    print("="*50 + "\n")
    
    return summary

if __name__ == "__main__":
    # Example execution for validation during development
    # Exemplo de execução para validação durante o desenvolvimento
    # Path to a test file (ensure this file exists or use 'raw' input)
    # Caminho para um arquivo de teste (garanta que o arquivo exista)
    TEST_FILE = "docs/example_report.txt" 
    
    if os.path.exists(TEST_FILE):
        run_summarize_ai(TEST_FILE, mode="bullets")
    else:
        print(f"File {TEST_FILE} not found. Please add it to 'docs/' directory.")