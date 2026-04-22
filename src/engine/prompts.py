from langchain.prompts import PromptTemplate

# 3. CUSTOM PROMPT TEMPLATES FOR DIFFERENT MODES
# 3. TEMPLATES DE PROMPT CUSTOMIZADOS PARA DIFERENTES MODOS

class PromptFactory:
    """
    Factory to manage different summarization styles and formats.
    Factory para gerenciar diferentes estilos e formatos de sumarização.
    """

    @staticmethod
    def get_map_prompt() -> PromptTemplate:
        """
        Standard prompt to summarize each individual text chunk.
        Prompt padrão para sumarizar cada pedaço de texto individual.
        """
        template = """
        Write a concise summary of the following text, preserving key technical terms:
        "{text}"
        CONCISE SUMMARY:
        """
        return PromptTemplate.from_template(template)

    @staticmethod
    def get_reduce_prompt(mode: str = "executive") -> PromptTemplate:
        """
        Returns a specific consolidation prompt based on the chosen mode.
        Retorna um prompt de consolidação específico baseado no modo escolhido.
        """
        
        # Mode: Short Summary (One paragraph) / Resumo Curto (Um parágrafo)
        if mode == "short":
            template = """
            Synthesize the following points into a single, professional paragraph:
            "{text}"
            SHORT SUMMARY:
            """
        
        # Mode: Bullet Points (Action oriented) / Pontos Chave (Focado em ação)
        elif mode == "bullets":
            template = """
            Based on the provided summaries, extract the main decisions and action points as bullet points:
            "{text}"
            KEY ACTION POINTS:
            """
            
        # Mode: Executive Briefing (Structured) / Resumo Executivo (Estruturado)
        else:
            template = """
            Create a structured executive briefing from the summaries below. 
            Include sections for 'Introduction', 'Main Findings', and 'Conclusion':
            "{text}"
            EXECUTIVE BRIEFING:
            """
            
        return PromptTemplate.from_template(template)

# Engineering Note: Prompt engineering is key to avoiding LLM hallucinations.
# Nota de Engenharia: A engenharia de prompt é fundamental para evitar alucinações da LLM.