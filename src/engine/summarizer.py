from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import HuggingFaceHub
from langchain_core.documents import Document

# 3. SUMMARIZATION ENGINE PIPELINE
# 3. PIPELINE DO MOTOR DE SUMARIZAÇÃO

class SummarizationEngine:
    """
    Class responsible for executing the summarization logic using LangChain.
    Classe responsável por executar a lógica de sumarização usando LangChain.
    """

    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initializes the LLM and the summarization chain.
        Inicializa o LLM e a cadeia de sumarização.
        """
        # Initializing Open-Source LLM via HuggingFace
        # Inicializando LLM Open-Source via HuggingFace
        self.llm = HuggingFaceHub(
            repo_id=model_id,
            model_kwargs={"temperature": 0.3, "max_new_tokens": 1024},
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )

    def generate_summary(self, chunks: list[Document], chain_type: str = "map_reduce") -> str:
        """
        Executes the summarization chain (Map-Reduce) on text chunks.
        Executa a cadeia de sumarização (Map-Reduce) nos pedaços de texto.
        """
        if not chunks:
            return "No content to summarize. / Nenhum conteúdo para sumarizar."

        # load_summarize_chain manages the prompt logic for Map and Reduce phases
        # load_summarize_chain gerencia a lógica de prompts para as fases de Map e Reduce
        summary_chain = load_summarize_chain(
            llm=self.llm, 
            chain_type=chain_type,
            verbose=False # Set to True for debugging / Defina como True para depuração
        )

        try:
            # Running the chain / Executando a cadeia
            output = summary_chain.invoke(chunks)
            return output["output_text"]
        except Exception as e:
            return f"Error during summarization: {e} / Erro durante a sumarização: {e}"

# Integration Test / Teste de Integração
if __name__ == "__main__":
    # This represents the flow from previous phases
    # Isso representa o fluxo das fases anteriores
    engine = SummarizationEngine()
    print("Summarizer initialized successfully. / Sumarizador inicializado com sucesso.")