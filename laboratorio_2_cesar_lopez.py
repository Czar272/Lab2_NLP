"""
CC3103 - Procesamiento de Lenguaje Natural
Laboratorio 2: Motor de Busqueda Semantica

Autor: Cesar Lopez

Descripcion:
    Motor de busqueda semantica simple sobre un corpus de soporte tecnico.
    - Genera embeddings de oraciones con un modelo local de Hugging Face.
    - Calcula similitud coseno entre una consulta y el corpus.
    - Devuelve los resultados top-k.
    - Compara la busqueda semantica contra una busqueda por palabras clave.
    - Evalua ambos metodos con un pequeno conjunto de juicios de relevancia.

Instalacion requerida:
    pip install sentence-transformers numpy scikit-learn

Ejecutar:
    python laboratorio_2_cesar_lopez.py
"""

from __future__ import annotations

from typing import Iterable
import re
import sys

import numpy as np


MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3


# -----------------------------------------------------------------------------
# 1. Corpus De Prueba (Parte A)
# -----------------------------------------------------------------------------

# Dominio: soporte tecnico de una plataforma en linea (cuentas, acceso,
# correo, aplicacion movil, pagos, conectividad y hardware).
CORPUS = [
    "No puedo iniciar sesión en mi cuenta.",                                   # 0
    "Olvidé mi clave de acceso al sistema.",                                   # 1
    "El usuario desea cambiar su contraseña.",                                 # 2
    "La plataforma muestra un error al iniciar sesión.",                       # 3
    "El cliente quiere actualizar su correo electrónico.",                     # 4
    "La factura fue enviada al correo registrado.",                            # 5
    "El sistema recomienda restablecer la clave.",                             # 6
    "La aplicación móvil no carga correctamente.",                             # 7
    "La cuenta fue bloqueada por demasiados intentos fallidos.",               # 8
    "Necesito recuperar el acceso porque perdí mis credenciales.",             # 9
    "No recibo el código de verificación por mensaje de texto.",               # 10
    "La app se cierra sola cada vez que la abro en el teléfono.",              # 11
    "El cobro de la suscripción aparece duplicado en mi tarjeta.",             # 12
    "Quiero cancelar mi plan y solicitar un reembolso.",                       # 13
    "La conexión a internet es muy lenta en la oficina.",                      # 14
    "El router se reinicia constantemente y pierdo la señal wifi.",            # 15
    "La impresora no imprime los documentos enviados.",                        # 16
    "La computadora se apaga sola después de unos minutos.",                   # 17
    "Al abrir el programa aparece una pantalla azul.",                         # 18
    "No puedo descargar los archivos adjuntos del correo.",                    # 19
    "Los mensajes nuevos no llegan a mi bandeja de entrada.",                  # 20
    "Deseo modificar el número de teléfono asociado a mi perfil.",             # 21
    "La autenticación de dos factores no acepta el código.",                   # 22
    "El servidor no responde y la página tarda mucho en cargar.",              # 23
    "Necesito instalar la última actualización del software.",                 # 24
]


# -----------------------------------------------------------------------------
# 2. Consultas De Prueba (Parte B)
# -----------------------------------------------------------------------------

# Cada consulta incluye el tipo de caso que pretende evaluar y los indices del
# corpus que consideramos relevantes (juicio manual) para medir los resultados.
CONSULTAS_ETIQUETADAS = [
    {
        "consulta": "problemas con mi contraseña",
        "tipo": "coincidencia parcial de palabras (contraseña) + sinonimos (clave)",
        "relevantes": [1, 2, 6, 9],
    },
    {
        "consulta": "no puedo entrar a mi cuenta",
        "tipo": "coincidencia directa de palabras",
        "relevantes": [0, 3, 8, 9],
    },
    {
        "consulta": "quiero cambiar mi email",
        "tipo": "sin coincidencia directa, relacion semantica (email = correo)",
        "relevantes": [4],
    },
    {
        "consulta": "el celular no abre la aplicación",
        "tipo": "sinonimos (celular = telefono / movil)",
        "relevantes": [7, 11],
    },
    {
        "consulta": "me cobraron dos veces",
        "tipo": "sin palabras compartidas, relacion semantica",
        "relevantes": [12],
    },
    {
        "consulta": "el equipo no funciona",
        "tipo": "consulta ambigua (equipo = computadora, router, impresora...)",
        "relevantes": [15, 16, 17, 18],
    },
    {
        "consulta": "no tengo internet",
        "tipo": "palabras clave pueden fallar (solo comparte 'no' / 'internet')",
        "relevantes": [14, 15],
    },
    {
        "consulta": "no me llega el SMS para entrar",
        "tipo": "abreviatura / jerga (SMS = mensaje de texto)",
        "relevantes": [10, 22],
    },
]

CONSULTAS = [item["consulta"] for item in CONSULTAS_ETIQUETADAS]


# -----------------------------------------------------------------------------
# 3. Utilidades De Instalacion
# -----------------------------------------------------------------------------

def cargar_modelo(nombre_modelo: str):
    """Carga un modelo de sentence-transformers.

    Si la dependencia no esta instalada, muestra una instruccion clara.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        mensaje = (
            "No se encontro la libreria 'sentence-transformers'.\n"
            "Instale las dependencias con:\n\n"
            "    pip install sentence-transformers numpy scikit-learn\n\n"
            "Si usa Google Colab:\n\n"
            "    !pip install sentence-transformers numpy scikit-learn\n"
        )
        raise SystemExit(mensaje) from exc

    return SentenceTransformer(nombre_modelo)


# -----------------------------------------------------------------------------
# 4. Embeddings (Parte C)
# -----------------------------------------------------------------------------

def generar_embeddings(modelo, textos: list[str]) -> np.ndarray:
    """Genera embeddings normalizados (norma L2 = 1) para una lista de textos.

    Con vectores normalizados, el producto punto es exactamente la
    similitud coseno.
    """
    embeddings = modelo.encode(textos, normalize_embeddings=True)
    return np.asarray(embeddings)


# -----------------------------------------------------------------------------
# 5. Similitud Coseno Y Ranking (Partes D y E)
# -----------------------------------------------------------------------------

def similitud_coseno(embedding_consulta: np.ndarray, embeddings_corpus: np.ndarray) -> np.ndarray:
    """Calcula la similitud coseno entre una consulta y todos los documentos.

    cos(q, d) = (q . d) / (||q|| * ||d||)

    Se calcula de forma explicita (no se asume normalizacion previa), por lo
    que la funcion es correcta aunque los embeddings no esten normalizados.
    """
    norma_consulta = np.linalg.norm(embedding_consulta)
    normas_corpus = np.linalg.norm(embeddings_corpus, axis=1)
    denominador = np.clip(normas_corpus * norma_consulta, 1e-12, None)
    return (embeddings_corpus @ embedding_consulta) / denominador


def recuperar_top_k(puntajes: np.ndarray, corpus: list[str], top_k: int) -> list[dict]:
    """Ordena los puntajes de mayor a menor y devuelve los k mejores.

    Cada resultado incluye: posicion en el ranking, indice original en el
    corpus, texto recuperado y puntaje de similitud.
    """
    # Orden estable: ante empates se conserva el orden original del corpus.
    indices_ordenados = np.argsort(-puntajes, kind="stable")[:top_k]

    return [
        {
            "rank": posicion,
            "indice": int(indice),
            "texto": corpus[indice],
            "score": float(puntajes[indice]),
        }
        for posicion, indice in enumerate(indices_ordenados, start=1)
    ]


def buscar_semanticamente(
    consulta: str,
    corpus: list[str],
    embeddings_corpus: np.ndarray,
    modelo,
    top_k: int = TOP_K,
) -> list[dict]:
    """Devuelve los top-k textos semanticamente mas similares a la consulta."""
    embedding_consulta = generar_embeddings(modelo, [consulta])[0]
    puntajes = similitud_coseno(embedding_consulta, embeddings_corpus)
    return recuperar_top_k(puntajes, corpus, top_k)


# -----------------------------------------------------------------------------
# 6. Busqueda Por Palabras Clave (Parte F)
# -----------------------------------------------------------------------------

def tokenizar_simple(texto: str) -> set[str]:
    """Tokeniza texto: minusculas y extraccion de palabras alfanumericas."""
    return set(re.findall(r"\b\w+\b", texto.lower()))


def buscar_por_palabras_clave(consulta: str, corpus: list[str], top_k: int = TOP_K) -> list[dict]:
    """Busca textos por cantidad de palabras compartidas con la consulta.

    Esta implementacion es intencionalmente simple (sin stemming, sin
    eliminacion de stopwords, sin normalizar tildes) para servir como linea
    base frente a la busqueda semantica.
    """
    tokens_consulta = tokenizar_simple(consulta)
    resultados = []

    for indice, texto in enumerate(corpus):
        tokens_texto = tokenizar_simple(texto)
        coincidencias = tokens_consulta.intersection(tokens_texto)

        resultados.append({
            "indice": indice,
            "texto": texto,
            "score": len(coincidencias),
            "coincidencias": sorted(coincidencias),
        })

    # sort es estable: ante empates se conserva el orden del corpus.
    resultados.sort(key=lambda item: item["score"], reverse=True)

    for posicion, resultado in enumerate(resultados, start=1):
        resultado["rank"] = posicion

    return resultados[:top_k]


# -----------------------------------------------------------------------------
# 7. Impresion De Resultados (Parte G)
# -----------------------------------------------------------------------------

def imprimir_resultados_semanticos(resultados: Iterable[dict]) -> None:
    print("\nBusqueda semantica")
    for r in resultados:
        print(f"  {r['rank']}. score={r['score']:.4f} | idx={r['indice']:>2} | {r['texto']}")


def imprimir_resultados_keyword(resultados: Iterable[dict]) -> None:
    print("\nBusqueda por palabras clave")
    for r in resultados:
        coincidencias = ", ".join(r["coincidencias"]) or "sin coincidencias"
        print(
            f"  {r['rank']}. score={r['score']} | idx={r['indice']:>2} | "
            f"coincidencias={coincidencias} | {r['texto']}"
        )


def comparar_busquedas(
    consulta: str,
    corpus: list[str],
    embeddings_corpus: np.ndarray,
    modelo,
    top_k: int = TOP_K,
    tipo: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Imprime busqueda semantica y keyword search para una consulta."""
    print("=" * 100)
    print(f"CONSULTA: {consulta}")
    if tipo:
        print(f"Tipo de caso: {tipo}")
    print("=" * 100)

    resultados_semanticos = buscar_semanticamente(
        consulta=consulta,
        corpus=corpus,
        embeddings_corpus=embeddings_corpus,
        modelo=modelo,
        top_k=top_k,
    )
    resultados_keyword = buscar_por_palabras_clave(consulta, corpus, top_k=top_k)

    imprimir_resultados_semanticos(resultados_semanticos)
    imprimir_resultados_keyword(resultados_keyword)
    print()

    return resultados_semanticos, resultados_keyword


# -----------------------------------------------------------------------------
# 8. Evaluacion Simple (apoyo para la reflexion)
# -----------------------------------------------------------------------------

def precision_en_k(resultados: list[dict], relevantes: set[int]) -> float:
    """Proporcion de los resultados devueltos que son relevantes."""
    if not resultados:
        return 0.0
    aciertos = sum(1 for r in resultados if r["indice"] in relevantes)
    return aciertos / len(resultados)


def rango_reciproco(resultados: list[dict], relevantes: set[int]) -> float:
    """1 / posicion del primer resultado relevante (0 si no hay ninguno)."""
    for r in resultados:
        if r["indice"] in relevantes:
            return 1.0 / r["rank"]
    return 0.0


def filtrar_keyword_con_coincidencias(resultados: list[dict]) -> list[dict]:
    """Un resultado keyword con score=0 no es una recuperacion real."""
    return [r for r in resultados if r["score"] > 0]


def imprimir_evaluacion(evaluaciones: list[dict], top_k: int) -> None:
    print("=" * 100)
    print(f"EVALUACION (P@{top_k} y MRR con juicios de relevancia manuales)")
    print("=" * 100)
    print(f"{'Consulta':<36} | {'Sem P@k':>7} | {'Sem RR':>6} | {'Kw P@k':>6} | {'Kw RR':>5}")
    print("-" * 75)

    for e in evaluaciones:
        print(
            f"{e['consulta']:<36} | {e['sem_p']:>7.2f} | {e['sem_rr']:>6.2f} | "
            f"{e['kw_p']:>6.2f} | {e['kw_rr']:>5.2f}"
        )

    print("-" * 75)
    n = len(evaluaciones)
    print(
        f"{'PROMEDIO':<36} | "
        f"{sum(e['sem_p'] for e in evaluaciones) / n:>7.2f} | "
        f"{sum(e['sem_rr'] for e in evaluaciones) / n:>6.2f} | "
        f"{sum(e['kw_p'] for e in evaluaciones) / n:>6.2f} | "
        f"{sum(e['kw_rr'] for e in evaluaciones) / n:>5.2f}"
    )
    print("\nNota: en keyword search los resultados con score=0 se cuentan como no relevantes.")


# -----------------------------------------------------------------------------
# 9. Programa Principal
# -----------------------------------------------------------------------------

def main() -> None:
    # Evita errores de codificacion con tildes y enes en consolas de Windows.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Cargando modelo de embeddings...")
    modelo = cargar_modelo(MODELO_EMBEDDINGS)

    print("Generando embeddings del corpus...")
    embeddings_corpus = generar_embeddings(modelo, CORPUS)

    print(f"Corpus: {len(CORPUS)} oraciones")
    print(f"Dimension de embeddings: {embeddings_corpus.shape[1]}")
    print(f"Consultas: {len(CONSULTAS)}\n")

    evaluaciones = []
    for item in CONSULTAS_ETIQUETADAS:
        semanticos, keyword = comparar_busquedas(
            consulta=item["consulta"],
            corpus=CORPUS,
            embeddings_corpus=embeddings_corpus,
            modelo=modelo,
            top_k=TOP_K,
            tipo=item["tipo"],
        )

        relevantes = set(item["relevantes"])
        keyword_validos = filtrar_keyword_con_coincidencias(keyword)
        evaluaciones.append({
            "consulta": item["consulta"],
            "sem_p": precision_en_k(semanticos, relevantes),
            "sem_rr": rango_reciproco(semanticos, relevantes),
            "kw_p": sum(1 for r in keyword_validos if r["indice"] in relevantes) / TOP_K,
            "kw_rr": rango_reciproco(keyword_validos, relevantes),
        })

    imprimir_evaluacion(evaluaciones, TOP_K)


if __name__ == "__main__":
    main()
