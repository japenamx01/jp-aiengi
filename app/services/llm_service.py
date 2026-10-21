from openai import OpenAI
from app.config import settings
from app.context.examples import ESTIMATION_EXAMPLES

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def generate_estimation(transcription: str) -> str:
    # Construcción de los ejemplos de manera explícita sin bucles complejos en línea
    examples_list = []
    for ex in ESTIMATION_EXAMPLES:
        example_str = f"Ejemplo de Reunión:\n{ex['meeting_summary']}\n\nEstimación esperada:\n{ex['estimation']}"
        examples_list.append(example_str)
    
    examples_text = "\n\n".join(examples_list)

    system_prompt = f"""
    Eres un estimador de software experto. Tu labor es analizar la transcripción de una reunión con un cliente y generar una estimación estructurada de desarrollo.
    Basate estrictamente en el formato y nivel de detalle de los siguientes ejemplos históricos de referencia:

    {examples_text}
    """

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transcripción de la nueva reunión a estimar:\n{transcription}"}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

