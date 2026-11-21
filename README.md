# Laboratorio 2 - Motor de Búsqueda Semántica

CC3103 - Procesamiento de Lenguaje Natural
Autor: César López

Motor de búsqueda semántica simple sobre un corpus de **soporte técnico**, usando embeddings locales de
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, comparado contra una búsqueda básica por
palabras clave.

## Archivos

| Archivo | Contenido |
|---|---|
| `laboratorio_2_cesar_lopez.py` | Implementación completa (corpus, consultas, embeddings, coseno, top-k, keyword search, comparación y evaluación). |
| `resultados.txt` | Salida completa de la ejecución del programa. |
| `requirements.txt` | Dependencias. |

## Ejecución

```bash
pip install -r requirements.txt
python laboratorio_2_cesar_lopez.py
```

La primera ejecución descarga el modelo (~470 MB) desde Hugging Face; luego funciona en CPU sin conexión.

## Pipeline

```text
Corpus (25 oraciones) -> embeddings normalizados (384 dims)
  -> consulta -> embedding de la consulta
  -> similitud coseno -> ranking top-k (k=3)
  -> comparación con keyword search (intersección de tokens)
```

- **Parte A - Corpus:** 25 oraciones en español sobre soporte técnico (acceso, contraseñas, correo, app móvil,
  pagos, conectividad, hardware).
- **Parte B - Consultas:** 8 consultas que cubren coincidencia directa, sinónimos sin palabras compartidas,
  una consulta ambigua y casos donde keyword search falla. Cada una tiene juicios de relevancia manuales.
- **Parte C - Embeddings:** `modelo.encode(..., normalize_embeddings=True)`; se imprime el tamaño del corpus
  (25) y la dimensión (384).
- **Parte D - Similitud coseno:** `cos(q, d) = q·d / (‖q‖‖d‖)`, calculada explícitamente.
- **Parte E - Top-k:** cada resultado incluye posición, índice original, texto y puntaje.
- **Parte F - Palabras clave:** tokenización en minúsculas con `\b\w+\b`, conteo de palabras compartidas y
  orden por número de coincidencias.
- **Parte G - Comparación:** para cada consulta se imprimen ambos rankings (ver `resultados.txt`).
- **Extra - Evaluación:** Precision@3 y rango recíproco (MRR) por consulta con los juicios de relevancia.

## Resumen de resultados

| Consulta | Sem P@3 | Sem RR | Kw P@3 | Kw RR |
|---|---:|---:|---:|---:|
| problemas con mi contraseña | 0.67 | 1.00 | 0.67 | 0.50 |
| no puedo entrar a mi cuenta | 0.67 | 1.00 | 0.33 | 1.00 |
| quiero cambiar mi email | 0.33 | 1.00 | 0.00 | 0.00 |
| el celular no abre la aplicación | 0.67 | 1.00 | 0.33 | 1.00 |
| me cobraron dos veces | 0.33 | 1.00 | 0.00 | 0.00 |
| el equipo no funciona | 0.00 | 0.00 | 0.00 | 0.00 |
| no tengo internet | 0.33 | 0.33 | 0.00 | 0.00 |
| no me llega el SMS para entrar | 0.33 | 0.50 | 0.67 | 1.00 |
| **Promedio** | **0.42** | **0.73** | **0.25** | **0.44** |

En keyword search los resultados con `score=0` se cuentan como no relevantes.

## Reflexión

La búsqueda semántica funcionó mejor en "quiero cambiar mi email" y "me cobraron dos veces". En la primera
recuperó "El cliente quiere actualizar su correo electrónico" (0.78) sin compartir ninguna palabra de
contenido, mientras que keyword search devolvió "Quiero cancelar mi plan" solo por coincidir en "quiero" y
"mi". En la segunda, el modelo relacionó "cobraron dos veces" con "cobro duplicado"; la
línea base eligió la oración de autenticación "de dos factores".

La búsqueda por palabras clave funcionó mejor en "no me llega el SMS para entrar": puso primero "No recibo el
código de verificación por mensaje de texto". Fue suerte: coincidió con "el" y "no",
palabras vacías que también empataron con oraciones irrelevantes.

Lo inesperado fue "no tengo internet": la semántica puso primero "perdí mis credenciales" y "olvidé mi clave",
dejando la oración sobre internet en tercer lugar. El modelo parece interpretar "no tengo" como pérdida de
acceso. Además, la consulta ambigua "el equipo no funciona" falló en ambos métodos: "equipo" no se asoció con
computadora, router ni impresora.

Como limitaciones, el modelo da puntajes bajos y comprimidos (0.27-0.48) en consultas cortas o ambiguas,
confunde negaciones y prioriza estructura sintáctica ("no llegan a mi bandeja de entrada" frente a "no me
llega el SMS para entrar").

Para mejorar, agregaría más oraciones por tema con sinónimos y jerga (SMS, celular, equipo), eliminaría
palabras vacías y normalizaría tildes en keyword search, usaría BM25 y combinaría ambos métodos en una
búsqueda híbrida con un umbral mínimo de similitud.
