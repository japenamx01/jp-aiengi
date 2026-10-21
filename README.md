# Estimador de Software con Arquitectura CAG (Cache Augmented Generation)

Aplicación backend construida con **FastAPI** y **uv** que implementa una arquitectura **CAG**. El servicio recibe transcripciones de reuniones con clientes y genera estimaciones estructuradas de desarrollo utilizando un LLM con contexto estático inyectado directamente en el prompt.

---

## 🏗 Arquitectura del Proyecto

Este servicio utiliza la arquitectura **CAG** (Cache Augmented Generation). A diferencia de RAG tradicional, todo el contexto necesario (ejemplos históricos de estimaciones de referencia) viaja estáticamente en cada llamada al sistema, eliminando la necesidad de bases de datos vectoriales o sistemas de recuperación complejos para este caso de uso.

```text
estimador-cag/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── routers/
│   │   ├── __init__.py
│   │   └── estimations.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── llm_service.py
│   └── context/
│       ├── __init__.py
│       └── examples.py
├── .env
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md

