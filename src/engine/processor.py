from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 2. TEXT PROCESSING AND CHUNKING STRATEGY
# 2. ESTRATÉGIA DE PROCESSAMENTO DE TEXTO E DIVISÃO (CHUNKING)

class TextProcessor:
    """
    Class responsible for splitting long documents into manageable chunks for the LLM.
    Classe responsável por dividir documentos longos em pedaços gerenciáveis para o LLM.
    """

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        """
        Initializes the splitter with defined size and overlap.
        Inicializa o divisor com tamanho e sobreposição definidos.
        """
        # Recursive splitter tries to keep paragraphs and sentences together
        # O divisor recursivo tenta manter parágrafos e frases juntos
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_documents(self, documents: list[Document]) -> list[Document]:
        """
        Splits documents into smaller chunks while preserving metadata.
        Divide documentos em pedaços menores preservando os metadados.
        """
        if not documents:
            # Safety check for empty document lists
            # Verificação de segurança para listas de documentos vazias
            return []
            
        chunks = self.splitter.split_documents(documents)
        
        print(f"Status: {len(documents)} doc(s) split into {len(chunks)} chunks.")
        # Status: {len(documents)} doc(s) divididos em {len(chunks)} pedaços.
        
        return chunks

# Execution example for validation
# Exemplo de execução para validação
if __name__ == "__main__":
    processor = TextProcessor()
    sample_doc = [Document(page_content="Validation text content... " * 100)]
    final_chunks = processor.process_documents(sample_doc)
    print("Process finished successfully. / Processo finalizado com sucesso.")