from langchain.prompts import PromptTemplate

# 4. CUSTOM PROMPT TEMPLATES FOR DIFFERENT MODES
# 4. TEMPLATES DE PROMPT CUSTOMIZADOS PARA DIFERENTES MODOS

class PromptFactory:
    """
    Factory to manage different summarization styles.
    Factory para gerenciar diferentes estilos de sumarização.
    """

    @staticmethod
    def get_map_prompt():
        """
        Standard prompt to summarize each individual chunk.
        Prompt padrão para sumarizar cada pedaço individual.
        """
        template = """
        Write a concise summary of the following text:
        "{text}"
        CONCISE SUMMARY:
        """
        return PromptTemplate.from_template(template)

    @staticmethod
    def get_reduce_prompt(mode: str = "concise"):
        """
        Returns a specific prompt based on the user's chosen mode.
        Retorna um prompt específico com base no modo escolhido pelo usuário.
        """
        
        # Mode: Short Summary / Resumo Curto
        if mode == "short":
            template = """
            Write a final professional summary of the following summarized points in exactly one paragraph:
            "{text}"
            SHORT SUMMARY:
            """
        
        # Mode: Bullet Points / Pontos Chave
        elif mode == "bullets":
            template = """
            Based on the following summaries, extract the main decisions and key action points in bullet points:
            "{text}"
            KEY POINTS:
            """
            
        # Mode: Executive Briefing (Detailed) / Resumo Executivo (Detalhado)
        else:
            template = """
            Write a detailed executive briefing based on the following summaries. 
            Include an introduction, main findings, and a conclusion:
            "{text}"
            EXECUTIVE BRIEFING:
            """
            
        return PromptTemplate.from_template(template)

# Engineering Note: These templates ensure consistent output formatting
# Nota de Engenharia: Estes templates garantem formatação de saída consistente