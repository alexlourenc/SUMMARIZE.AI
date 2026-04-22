from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

# 1. SETUP AND INGESTION CLASS
# 1. CLASSE DE CONFIGURAÇÃO E INGESTÃO

class DataIngestion:
    """
    Class responsible for loading data from different sources (PDF, TXT, or Raw Text).
    Classe responsável por carregar dados de diferentes fontes (PDF, TXT ou Texto Bruto).
    """
    
    def __init__(self):
        # Initializer kept simple for future configuration expansions
        # Inicializador mantido simples para futuras expansões de configuração
        pass

    def load_data(self, source: str, source_type: str = "text") -> list[Document]:
        """
        Loads documents and converts them into LangChain Document objects.
        Carrega documentos e os converte em objetos Document do LangChain.
        """
        try:
            # Handles PDF files using page-by-page extraction
            # Gerencia arquivos PDF usando extração página por página
            if source_type == "pdf":
                loader = PyPDFLoader(source)
                return loader.load()
            
            # Handles standard text files
            # Gerencia arquivos de texto padrão
            elif source_type == "txt":
                loader = TextLoader(source)
                return loader.load()
            
            # Handles direct string input as a Document object
            # Gerencia entrada direta de string como um objeto Document
            elif source_type == "raw":
                return [Document(page_content=source, metadata={"source": "manual_input"})]
            
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
        
        except Exception as e:
            # Error logging for debugging purposes
            # Registro de erro para fins de depuração
            print(f"Error loading data: {e} / Erro ao carregar dados: {e}")
            return []

# Example of execution for validation
# Exemplo de execução para validação
if __name__ == "__main__":
    ingestor = DataIngestion()
    test_data = ingestor.load_data("Sample content for validation.", "raw")
    print(f"Execution successful: {len(test_data)} document(s) loaded.")