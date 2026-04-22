import os
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import HuggingFaceHub
from langchain_core.documents import Document
from src.engine.prompts import PromptFactory

# 4. SUMMARIZATION ENGINE PIPELINE
# 4. PIPELINE DO MOTOR DE SUMARIZAÇÃO

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
        # Securely retrieving the API Token from environment variables
        # Recuperando o Token da API com segurança das variáveis de ambiente
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        if not api_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found. / Token não encontrado.")

        # Initializing the Open-Source LLM
        # Inicializando o LLM Open-Source
        self.llm = HuggingFaceHub(
            repo_id=model_id,
            model_kwargs={"temperature": 0.3, "max_new_tokens": 1024},
            huggingfacehub_api_token=api_token
        )

    def generate_summary(self, chunks: list[Document], mode: str = "executive") -> str:
        """
        Executes the Map-Reduce chain using custom prompts based on the mode.
        Executa a cadeia Map-Reduce usando prompts customizados baseados no modo.
        """
        if not chunks:
            return "No content to summarize. / Nenhum conteúdo para sumarizar."

        # Fetching custom prompts from our PromptFactory
        # Buscando prompts customizados da nossa PromptFactory
        map_prompt = PromptFactory.get_map_prompt()
        reduce_prompt = PromptFactory.get_reduce_prompt(mode=mode)

        # Configuring the chain with the Map-Reduce strategy
        # Configurando a cadeia com a estratégia Map-Reduce
        summary_chain = load_summarize_chain(
            llm=self.llm, 
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=reduce_prompt,
            verbose=False
        )

        try:
            # Running the process and returning the consolidated text
            # Executando o processo e retornando o texto consolidado
            output = summary_chain.invoke(chunks)
            return output["output_text"]
        except Exception as e:
            return f"Error during summarization: {e} / Erro durante a sumarização: {e}"

# Integration Test / Teste de Integração
if __name__ == "__main__":
    try:
        engine = SummarizationEngine()
        print("Summarizer initialized successfully. / Sumarizador inicializado com sucesso.")
    except Exception as e:
        print(f"Initialization failed: {e} / Falha na inicialização: {e}")