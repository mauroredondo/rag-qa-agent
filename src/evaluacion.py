"""Evaluacion de la busqueda (recuperacion).

Usa evaluacion/preguntas.json. Cada pregunta con respuesta trae una frase de
`evidencia` que aparece en el documento `fuente`. Un fragmento recuperado es un
ACIERTO si es de esa fuente y contiene la frase. Se mide:

- acierto@k: porcentaje de preguntas con algun acierto entre los k primeros.
- MRR: media de 1/posicion del primer acierto (1 si es el primero, 0,5 si el
  segundo...). Premia poner el fragmento correcto arriba.

Las preguntas sin respuesta no tienen fragmento correcto; se usan para ver si la
puntuacion de la busqueda permite distinguirlas de las que si tienen respuesta.

Uso:
    python -m src.evaluacion                    # compara los tres modos
    python -m src.evaluacion --modo hibrido --detalle
"""
import argparse
import json
import math
import re

from .config import RAIZ, cargar_config
from .embeddings import crear_embedder
from .indice import Indice


def _norm(texto):
    return re.sub(r"\s+", " ", texto).lower()


def cargar_preguntas(ruta=None):
    ruta = ruta or RAIZ / "evaluacion" / "preguntas.json"
    return json.loads(open(ruta, encoding="utf-8").read())


def es_acierto(trozo, pregunta):
    return trozo.fuente == pregunta["fuente"] and any(
        _norm(e) in _norm(trozo.texto) for e in pregunta["evidencia"])


def intervalo_wilson(aciertos, n, z=1.96):
    """Intervalo de confianza del 95 % para una proporcion (Wilson)."""
    if n == 0:
        return (0.0, 0.0)
    p = aciertos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centro - margen, centro + margen)


def auc(positivos, negativos):
    """Probabilidad de que un valor de `positivos` sea mayor que uno de `negativos`
    (0,5 = no distingue; 1 = los separa perfectamente)."""
    if not positivos or not negativos:
        return float("nan")
    total = sum((p > n) + 0.5 * (p == n) for p in positivos for n in negativos)
    return total / (len(positivos) * len(negativos))


def evaluar(indice, preguntas, k=5, modo="denso"):
    con = [p for p in preguntas if p["tipo"] == "con_respuesta"]
    sin = [p for p in preguntas if p["tipo"] == "sin_respuesta"]
    posiciones, detalle = [], []
    for p in con:
        resultados = indice.buscar(p["pregunta"], k, modo)
        pos = next((i for i, (t, _) in enumerate(resultados, 1) if es_acierto(t, p)), None)
        posiciones.append(pos)
        detalle.append((p, pos, resultados[0]))
    n = len(con)
    aciertos = lambda kk: sum(1 for x in posiciones if x is not None and x <= kk)
    top1_con = [indice.buscar(p["pregunta"], 1, modo)[0][1] for p in con]
    top1_sin = [indice.buscar(p["pregunta"], 1, modo)[0][1] for p in sin]
    return {
        "modo": modo, "n": n, "k": k,
        "acierto@1": aciertos(1) / n, "acierto@3": aciertos(3) / n,
        "acierto@k": aciertos(k) / n,
        "ic95@k": intervalo_wilson(aciertos(k), n),
        "mrr": sum(1 / x for x in posiciones if x) / n,
        "auc_sin_respuesta": auc(top1_con, top1_sin),
        "top1_con": top1_con, "top1_sin": top1_sin,
        "detalle": detalle,
    }


def construir_indice(config):
    config = {**config, "carpeta_documentos": RAIZ / config["carpeta_documentos"]}
    return Indice.construir(config, crear_embedder(config))


def imprimir(r, detalle=False):
    lo, hi = r["ic95@k"]
    print(f"{r['modo']:8s} acierto@1 {r['acierto@1']:.0%} | @3 {r['acierto@3']:.0%} | "
          f"@{r['k']} {r['acierto@k']:.0%} (IC95 {lo:.0%}-{hi:.0%}) | MRR {r['mrr']:.3f} | "
          f"AUC sin-respuesta {r['auc_sin_respuesta']:.2f}")
    if detalle:
        for p, pos, (t, s) in r["detalle"]:
            if pos != 1:
                marca = "NO ENCONTRADA" if pos is None else f"en posicion {pos}"
                print(f"   {p['id']} {marca}: {p['pregunta']}")
                print(f"        1.o recuperado: {t.fuente[:12]} p{t.pagina} [{s:.3f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo", choices=["denso", "bm25", "hibrido", "todos"], default="todos")
    parser.add_argument("--sin-prefijo", action="store_true", help="desactiva prefijo_fuente")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tamano", type=int)
    parser.add_argument("--solape", type=int)
    parser.add_argument("--separador", help="regex que marca el inicio de una seccion")
    parser.add_argument("--prefijo", action="store_true",
                        help="anteponer el nombre del documento al texto de cada fragmento")
    parser.add_argument("--detalle", action="store_true")
    args = parser.parse_args()
    config = cargar_config()
    if args.tamano:
        config["tamano_trozo"] = args.tamano
    if args.solape:
        config["solapamiento"] = args.solape
    if args.prefijo:
        config["prefijo_fuente"] = True
    if args.sin_prefijo:
        config["prefijo_fuente"] = False
    if args.separador:
        config["separador_estructura"] = args.separador
    indice = construir_indice(config)
    print(f"{len(indice.trozos)} fragmentos (tamano {config['tamano_trozo']}, "
          f"solape {config['solapamiento']}, estructura: {config.get('separador_estructura')}, "
          f"prefijo: {config.get('prefijo_fuente', False)})")
    modos = ["denso", "bm25", "hibrido"] if args.modo == "todos" else [args.modo]
    preguntas = cargar_preguntas()
    for modo in modos:
        imprimir(evaluar(indice, preguntas, args.k, modo), args.detalle)


if __name__ == "__main__":
    main()
