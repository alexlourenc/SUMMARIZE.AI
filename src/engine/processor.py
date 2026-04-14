from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 2. TEXT PROCESSING AND CHUNKING STRATEGY
# 2. ESTRATÉGIA DE PROCESSAMENTO DE TEXTO E DIVISÃO (CHUNKING)

class TextProcessor:
    """
    Class responsible for splitting long documents into manageable chunks.
    Classe responsável por dividir documentos longos em pedaços gerenciáveis.
    """

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        """
        Initializes the splitter with defined size and overlap.
        Inicializa o divisor com tamanho e sobreposição definidos.
        """
        # We use 2000 characters to balance context and LLM performance
        # Usamos 2000 caracteres para equilibrar contexto e performance do LLM
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_documents(self, documents: list[Document]) -> list[Document]:
        """
        Splits the loaded documents into smaller chunks.
        Divide os documentos carregados em pedaços menores.
        """
        if not documents:
            return []
            
        # The split_documents method ensures metadata is preserved across chunks
        # O método split_documents garante que os metadados sejam preservados nos pedaços
        chunks = self.splitter.split_documents(documents)
        
        print(f"Original docs: {len(documents)} -> Total chunks: {len(chunks)}")
        # Documentos originais: {len(documents)} -> Total de pedaços: {len(chunks)}
        
        return chunks

# Execution example for validation
# Exemplo de execução para validação
if __name__ == "__main__":
    processor = TextProcessor()
    # Dummy document for testing / Documento fictício para teste
    sample_doc = [Document(page_content="Long text... " * 500)]
    final_chunks = processor.process_documents(sample_doc)