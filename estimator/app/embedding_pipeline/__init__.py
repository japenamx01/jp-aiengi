"""Embedding pipeline (Session 7): budget JSON -> structural chunks -> OpenAI vectors.

NOTA: es el primer paso del CAG hacia RAG. Aquí solo se generan los vectores en
memoria y se devuelven por HTTP; guardarlos (pgvector) y buscar en ellos llega
en la Sesión 08.

    schemas.py  -> contrato de datos (entrada: presupuestos, salida: chunks + vectores)
    chunker.py  -> 1 componente del presupuesto = 1 chunk, con cabecera del padre
    embedder.py -> llama a text-embedding-3-small en lotes, con reintentos
    router.py   -> POST /embeddings/ingest: orquesta chunk -> embed -> stats
"""
