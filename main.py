import os
from src.engine.ingestion import DataIngestion
from src.engine.processor import TextProcessor
from src.engine.summarizer import SummarizationEngine
from src.engine.prompts import PromptFactory
from langchain.chains.summarize import load_summarize_chain

# 5. MAIN PIPELINE EXECUTION
# 5. EXECUÇÃO DO PIPELINE PRINCIPAL

def run_summarize_ai(file_path: str, mode: str = "executive"):
    """
    Orchestrates the full flow of SUMMARIZE.AI.
    Orquestra o fluxo completo do SUMMARIZE.AI.
    """
    
    # Initialize components
    # Inicializa componentes
    ingestor = DataIngestion()
    processor = TextProcessor(chunk_size=2000, chunk_overlap=200)
    engine = SummarizationEngine()
    
    print(f"--- Starting SUMMARIZE.AI | Mode: {mode.upper()} ---")
    
    # 1. Ingestion / Ingestão
    # Detects type based on extension / Detecta tipo com base na extensão
    source_type = "pdf" if file_path.endswith(".pdf") else "txt"
    docs = ingestor.load_data(file_path, source_type)
    
    # 2. Chunking / Processamento de pedaços
    chunks = processor.process_documents(docs)
    
    # 3. Prompt Selection / Seleção de Prompt
    map_p = PromptFactory.get_map_prompt()
    reduce_p = PromptFactory.get_reduce_prompt(mode=mode)
    
    # 4. Summarization / Sumarização
    # We build the chain with custom prompts / Construímos a cadeia com prompts customizados
    chain = load_summarize_chain(
        llm=engine.llm,
        chain_type="map_reduce",
        map_prompt=map_p,
        combine_prompt=reduce_p,
        verbose=False
    )
    
    print("Executing LLM Chains... / Executando Cadeias de LLM...")
    result = chain.invoke(chunks)
    
    # 5. Output / Saída
    print("\n--- FINAL SUMMARY / RESUMO FINAL ---")
    print(result["output_text"])
    
    return result["output_text"]

if __name__ == "__main__":
    # Ensure API Key is present / Garante que a chave da API está presente
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        print("Error: API Token missing. / Erro: Token de API ausente.")
    else:
        # Example call / Exemplo de chamada
        # run_summarize_ai("docs/project_report.pdf", mode="bullets")
        pass