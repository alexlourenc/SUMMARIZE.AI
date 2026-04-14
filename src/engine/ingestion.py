import os
from typing import Union
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

# 1. SETUP AND INGESTION CLASS
# 1. CLASSE DE CONFIGURAÇÃO E INGESTÃO

class DataIngestion:
    """
    Class responsible for loading data from different sources.
    Classe responsável por carregar dados de diferentes fontes.
    """
    
    def __init__(self):
        # Professional practice: ensure clear separation of concerns
        # Prática profissional: garantir separação clara de responsabilidades
        pass

    def load_data(self, source: str, source_type: str = "text") -> list[Document]:
        """
        Loads documents based on type: 'pdf', 'txt', or 'raw'.
        Carrega documentos com base no tipo: 'pdf', 'txt' ou 'raw'.
        """
        try:
            if source_type == "pdf":
                # PDF Loader handles page-by-page extraction
                # PDF Loader gerencia a extração página por página
                loader = PyPDFLoader(source)
                return loader.load()
            
            elif source_type == "txt":
                # TextLoader for standard unstructured files
                # TextLoader para arquivos não estruturados padrão
                loader = TextLoader(source)
                return loader.load()
            
            elif source_type == "raw":
                # Wraps raw string into a LangChain Document object
                # Envolve uma string bruta em um objeto Document do LangChain
                return [Document(page_content=source, metadata={"source": "manual_input"})]
            
            else:
                raise ValueError("Unsupported source type. / Tipo de fonte não suportado.")
        
        except Exception as e:
            print(f"Error loading data: {e} / Erro ao carregar dados: {e}")
            return []

# Example of usage following best practices
# Exemplo de uso seguindo as melhores práticas
if __name__ == "__main__":
    ingestor = DataIngestion()
    # Testing with a hypothetical raw text
    # Testando com um texto bruto hipotético
    test_data = ingestor.load_data("This is a long project report content...", "raw")
    print(f"Successfully loaded {len(test_data)} document object(s).")