"""Evaluacion del agente completo (no solo de la busqueda).

Para cada pregunta de evaluacion/preguntas.json ejecuta el agente y mide:

Preguntas CON respuesta
- respondida: el agente contesto en vez de rechazar.
- cita correcta: entre los fragmentos que cita hay uno con la evidencia esperada.
- rechazo erroneo: se nego a contestar teniendo la respuesta en los documentos.

Preguntas SIN respuesta
- rechazo correcto: se nego (no se invento nada).

Que la respuesta sea la correcta lo tiene que juzgar una persona: se guardan todas
en evaluacion/respuestas_agente.md junto a la respuesta esperada.

Uso:
    python -m src.evaluacion_agente
"""
import time

from .agente import construir_agente, preguntar
from .buscar import obtener_indice
from .config import RAIZ, cargar_config
from .evaluacion import cargar_preguntas, es_acierto, intervalo_wilson
from .llm import crear_llm


def evaluar_agente(agente, indice, preguntas, pausa=0.0):
    filas = []
    for p in preguntas:
        r = preguntar(agente, p["pregunta"])
        trozos_citados = [indice.trozos[f["trozo_id"]] for f in r["fuentes"]]
        filas.append({
            "pregunta": p, "resultado": r,
            "respondida": r["estado"] == "respondida",
            "cita_correcta": p["tipo"] == "con_respuesta"
                             and any(es_acierto(t, p) for t in trozos_citados),
            "busqueda_ok": p["tipo"] == "con_respuesta"
                           and any(es_acierto(indice.trozos[i], p) for i in r["recuperados"]),
        })
        time.sleep(pausa)
    return filas


def resumen(filas):
    con = [f for f in filas if f["pregunta"]["tipo"] == "con_respuesta"]
    sin = [f for f in filas if f["pregunta"]["tipo"] == "sin_respuesta"]
    cuenta = lambda xs, clave: sum(1 for f in xs if f[clave])
    return {
        "n_con": len(con), "n_sin": len(sin),
        "respondidas": cuenta(con, "respondida"),
        "cita_correcta": cuenta(con, "cita_correcta"),
        "busqueda_ok": cuenta(con, "busqueda_ok"),
        "rechazos_correctos": sum(1 for f in sin if not f["respondida"]),
    }


def _linea(nombre, k, n):
    lo, hi = intervalo_wilson(k, n)
    return f"{nombre}: {k}/{n} = {k / n:.0%} (IC95 {lo:.0%}-{hi:.0%})"


def imprimir(r):
    print(_linea("Preguntas con respuesta - la busqueda trae la evidencia", r["busqueda_ok"], r["n_con"]))
    print(_linea("Preguntas con respuesta - el agente responde", r["respondidas"], r["n_con"]))
    print(_linea("Preguntas con respuesta - cita el fragmento correcto", r["cita_correcta"], r["n_con"]))
    print(_linea("Preguntas sin respuesta - rechaza (no se inventa nada)", r["rechazos_correctos"], r["n_sin"]))


def guardar_respuestas(filas, ruta):
    lineas = ["# Respuestas del agente a las preguntas de evaluación", "",
              "Revisar a mano: ¿la respuesta del agente coincide con la esperada?", ""]
    for f in filas:
        p, r = f["pregunta"], f["resultado"]
        lineas += [f"## {p['id']}. {p['pregunta']}", "",
                   f"- **Esperada:** {p['respuesta_esperada'] or '(los documentos no la contienen: debe rechazar)'}",
                   f"- **Agente:** {r['respuesta']}"]
        if r["estado"] == "respondida":
            fuentes = ", ".join(f"{x['fuente']} p.{x['pagina']}" for x in r["fuentes"])
            lineas += [f"- **Cita:** «{r['cita_textual']}»", f"- **Fuentes:** {fuentes}",
                       f"- **Cita con la evidencia esperada:** {'sí' if f['cita_correcta'] else 'no'}"]
        else:
            lineas.append(f"- **Rechazada:** {r['motivo']}")
        lineas.append("")
    ruta.write_text("\n".join(lineas), encoding="utf-8")


def main():
    config = cargar_config()
    indice = obtener_indice(config)
    agente = construir_agente(indice, crear_llm(config), config)
    filas = evaluar_agente(agente, indice, cargar_preguntas())
    imprimir(resumen(filas))
    ruta = RAIZ / "evaluacion" / "respuestas_agente.md"
    guardar_respuestas(filas, ruta)
    print(f"\nRespuestas guardadas en {ruta.relative_to(RAIZ)} (modelo: {config['modelo_generacion']})")


if __name__ == "__main__":
    main()
