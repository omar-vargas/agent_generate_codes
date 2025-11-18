from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import List, Optional
from typing import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
import os
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from dotenv import load_dotenv
from openai import AzureOpenAI
import random
from typing import Literal
import ast
import re
import json
from fastapi.responses import JSONResponse
from datetime import datetime
import uuid

# Cargar las variables de entorno
load_dotenv()

# Establecer variables de entorno directamente
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")

print("✅ Variables de entorno cargadas correctamente.")

from typing import Annotated
import operator
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

prompt_agente_generador_codigos = {
    "role": "system",
    "content": "Eres un experto generador de códigos de evaluación cualitativa. Tu tarea consiste en analizar textos de diversas fuentes de datos y generar códigos que correspondan a las ideas o conceptos clave que surjan en los textos. El usuario te proporcionará los objetivos o hipotesis de la investigacion cualitativa  y la data de la investigacion de entrevistas para que los utilices para generar estos códigos. como respuesta debe dar solo la lista de codigos separadada por comas nada mas no añadas comentarios ni antes ni despues"
}

prompt_agente_generador_codigos_feedback = {
    "role": "system",
    "content": "Eres un experto generador de códigos de evaluación cualitativa. Tu tarea consiste en analizar textos de diversas fuentes de datos y generar códigos que correspondan a las ideas o conceptos clave que surjan en los textos. El usuario te proporcionará los codigos generados hasta ahora ,la data de la investigacion de entrevistas y por ultimo un feedback para que lo tengas en cuenta a la hora de generar nuevos codigos para que los utilices . como respuesta debe dar solo la lista de codigos separadada por comas nada mas no añadas comentarios ni antes ni despues"
}

prompt_agente_corrector_codigos = {
    "role": "system",
    "content": (
        "Eres un experto revisor de códigos de evaluación cualitativa. "
        "Dado un conjunto de datos cualitativos y unos códigos iniciales, "
        "tu tarea es validar o justificar los códigos comparándolos con la data. "
        "Devuelve una lista JSON de objetos con este formato exacto:\n\n"
        '[{"codigo": "nombre_del_codigo", "justificacion": "por qué es relevante", "ejemplos": "frases o fragmentos textuales"}]\n\n'
        "No devuelvas texto antes o después del JSON, ni explicaciones adicionales."
    )
}



prompt_agente_familias = {
    "role": "system",
    "content": (
        "Eres un experto en análisis cualitativo. "
        "Tu tarea es agrupar una lista de códigos en familias temáticas.\n\n"
        "REGLAS:\n"
        "- Debes usar SOLO los códigos proporcionados.\n"
        "- Cada código debe pertenecer a UNA SOLA familia.\n"
        "- No puede haber códigos repetidos en distintas familias.\n"
        "- El nombre de la familia puede ser un término nuevo que generalice los códigos.\n"
        "- Opcionalmente, puedes añadir una breve descripción de la familia.\n\n"
        "FORMATO DE RESPUESTA (JSON ESTRICTO):\n"
        "[\n"
        "  {\n"
        "    \"familia\": \"nombre_de_la_familia\",\n"
        "    \"descripcion\": \"breve descripción de la familia\",\n"
        "    \"codigos\": [\"codigo1\", \"codigo2\", \"codigo3\"]\n"
        "  }\n"
        "]\n\n"
        "No devuelvas ningún texto fuera del JSON."
    )
}




# Definir el tipo de estado
class State(TypedDict):
    data: Annotated[list[str], operator.add]
    questions: Annotated[list[str], operator.add]
    question: Annotated[list[str], operator.add]
    codes: Annotated[list[str], operator.add]
    final_codes: Annotated[list[str], operator.add]
    annotations: Annotated[list[str], operator.add]
    validate: bool
    temperature: NotRequired[float]
    max_tokens: NotRequired[int]
    top_p: NotRequired[float]
    frequency_penalty: NotRequired[float]
    presence_penalty: NotRequired[float]

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://invuniandesai-2.openai.azure.com/"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY", "DEFAULT_KEY"),
    api_version="2024-10-21"
)

def run_prompt(client, prompt_message: str, role_agent: dict, model: str = "gpt", temperature: float = 0.7, max_tokens: int = None, top_p: float = 1.0, frequency_penalty: float = 0.0, presence_penalty: float = 0.0, **kwargs):
    response = client.chat.completions.create(
        model=model,
        messages=[role_agent, {"role": "user", "content": prompt_message}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        **kwargs
    )
    
    tokens = response.usage.total_tokens
    content = response.choices[0].message.content

    return content, tokens

def call_azure_openai(prompt: str, option: int, temperature: float = 0.7, max_tokens: int = None, top_p: float = 1.0, frequency_penalty: float = 0.0, presence_penalty: float = 0.0, **kwargs) -> str:
    try:
        if option == 0:
            answer, tokens_used = run_prompt(client, prompt, prompt_agente_generador_codigos, temperature=temperature, max_tokens=max_tokens, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        elif option == 1:
            answer, tokens_used = run_prompt(client, prompt, prompt_agente_generador_codigos_feedback, temperature=temperature, max_tokens=max_tokens, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        elif option == 2:
            answer, tokens_used = run_prompt(client, prompt, prompt_agente_corrector_codigos, temperature=temperature, max_tokens=max_tokens, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        elif option == 3:
            answer, tokens_used = run_prompt(client, prompt, prompt_etiquetado, temperature=temperature, max_tokens=max_tokens, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        elif option == 4:
            # NUEVO: agrupación en familias
            answer, tokens_used = run_prompt(client, prompt, prompt_agente_familias, temperature=temperature, max_tokens=max_tokens, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        else:
            raise ValueError(f"Opción no válida: {option}")
        return answer
    except Exception as e:
        return f"Error al procesar la solicitud: {str(e)}"

def node_1(state: State) -> State:
    if "data" not in state:
        state["data"] = []
    if "questions" not in state:
        state["questions"] = []
    
    if state["questions"] and len(state["questions"]) > 0 and state["questions"][0].strip():
        prompt_text = 'La data es la siguiente ***' + str(state["data"]) + '***' + ' las preguntas son ***' + str(state["questions"])
    else:
        prompt_text = 'La data es la siguiente ***' + str(state["data"]) + '***. Analiza esta data cualitativa y genera códigos que correspondan a las ideas o conceptos clave que surjan en los textos.'
    
    temp = state.get("temperature", 0.7)
    max_tok = state.get("max_tokens")
    top_p = state.get("top_p", 1.0)
    freq_pen = state.get("frequency_penalty", 0.0)
    pres_pen = state.get("presence_penalty", 0.0)
    
    answer = call_azure_openai(prompt_text, 0, temperature=temp, max_tokens=max_tok, top_p=top_p, frequency_penalty=freq_pen, presence_penalty=pres_pen)
    
    state["codes"] = [answer]
    return state

def validate_data(state: State) -> State:
    if "data" not in state:
        state["data"] = []
    if "questions" not in state:
        state["questions"] = []
    if "annotations" not in state:
        state["annotations"] = []

    temp = state.get("temperature", 0.7)
    max_tok = state.get("max_tokens")
    top_p = state.get("top_p", 1.0)
    freq_pen = state.get("frequency_penalty", 0.0)
    pres_pen = state.get("presence_penalty", 0.0)

    answer = call_azure_openai(
        f'La data es la siguiente ***{str(state["data"])}*** los codigos generados fueron ***{str(state["codes"])}***',
        2,
        temperature=temp, max_tokens=max_tok, top_p=top_p, frequency_penalty=freq_pen, presence_penalty=pres_pen
    )

    state["annotations"] = [answer]
    result_storage["estado_actual"] = state
    return state

def human_aprobation(state: State, approved: bool = True, feedback: str = 'muchos codigos') -> Literal['node_1', END]:
    is_approved = interrupt(True)
    
    if state['validate'] == True:
        return END       
    else:
        return node_1

# Crear el grafo de estados
builder = StateGraph(State)
builder.add_node('node_1', node_1)
builder.add_node('validate_data', validate_data)
builder.add_edge(START, 'node_1')
builder.add_edge('node_1', 'validate_data')
builder.add_conditional_edges('validate_data', human_aprobation)

initial_state = {
    "data": ["percepcion de los estudiantes sobre la IA generatica"],
    "questions": ['los estudiantes cada ven pensaran menos y seran mas flojos']
}

# Configurar FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
result_storage = {}

# Modelos de datos para los endpoints
class InputData(BaseModel):
    texto: str
    preguntas: List[str]

class PreguntasRequest(BaseModel):
    preguntas: str
    session_id: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class ValidacionInput(BaseModel):
    aprobado: bool
    nuevos_codigos: List[str] = []
    feedback: str
    session_id: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class ConsolidatedFileRequest(BaseModel):
    content: str
    session_id: str


class FamiliasRequest(BaseModel):
    codigos_aprobados: List[str]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0



class Usuario(BaseModel):
    username: str
    password: str

def load_data_from_txt(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")
    
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read().strip() 
    
    initial_state["data"] = [content]

def run_graph_with_data(pregunta: str, session_id: str):
    print(pregunta)
    file_path = f"./consolidado_{session_id}.txt"
    load_data_from_txt(file_path)

    initial_state["questions"] = [pregunta]
    initial_state["validate"] = False

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    thread_config = {"configurable": {"thread_id": session_id}}

    result = graph.invoke(initial_state, thread_config)
    result_storage['annotations'] = result['annotations']
    return str(result['codes'])

def save_session_data(session_id: str, key: str, value):
    ruta = f"sessions/session_{session_id}.json"
    os.makedirs("sessions", exist_ok=True)
    data = {}
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
    data[key] = value
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_session_data(session_id: str):
    ruta = f"sessions/session_{session_id}.json"
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@app.post("/generar/")
def generar(request: PreguntasRequest):
    try:
        preguntas = request.preguntas
        session_id = request.session_id

        load_data_from_txt(f"./archivos_consolidados/consolidado_{session_id}.txt")
        
        results = []
        
        initial_state["questions"] = [preguntas] if preguntas.strip() else []
        initial_state["validate"] = False
        initial_state["temperature"] = request.temperature
        initial_state["max_tokens"] = request.max_tokens
        initial_state["top_p"] = request.top_p
        initial_state["frequency_penalty"] = request.frequency_penalty
        initial_state["presence_penalty"] = request.presence_penalty

        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)
        result = graph.invoke(initial_state, {"configurable": {"thread_id": session_id}})
        
        codes_str = result["codes"][0] if result["codes"] else ""
        codes_list = [c.strip() for c in codes_str.split(',') if c.strip()]
        results.append(codes_list)
        
        if preguntas.strip():
            state_without_hyp = {
                "data": initial_state["data"].copy(),
                "questions": [],
                "validate": False,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty
            }
            memory2 = MemorySaver()
            graph2 = builder.compile(checkpointer=memory2)
            result2 = graph2.invoke(state_without_hyp, {"configurable": {"thread_id": f"{session_id}_no_hyp"}})
            
            codes_str2 = result2["codes"][0] if result2["codes"] else ""
            codes_list2 = [c.strip() for c in codes_str2.split(',') if c.strip()]
            results.append(codes_list2)

        save_session_data(session_id, "codes", codes_list)
        save_session_data(session_id, "data", initial_state["data"])
        save_session_data(session_id, "questions", initial_state["questions"])
        
        if len(results) > 1:
            save_session_data(session_id, "codes_sin_hipotesis", codes_list2)

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/revisor/")
def obtener_codigos_primer_agente():
    try:
        if 'annotations' in result_storage:
            anotaciones = str(result_storage['annotations'])
            clean_str = anotaciones.strip()[1:-1]  

            indice_corte = clean_str.find('}\n{')  

            temas = clean_str[:indice_corte+1].replace("\n", "")
            justificaciones = clean_str[indice_corte+1:].replace("\n", "")

            return temas, justificaciones
        else:
            raise HTTPException(status_code=400, detail="No se han generado aún los códigos.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/guardar/")
async def guardar_archivo(request: ConsolidatedFileRequest):
    try:
        session_id = request.session_id
        os.makedirs("archivos_guardados", exist_ok=True)
        filename = f"output_{session_id}.txt"
        save_path = os.path.join("archivos_guardados", filename)

        with open(save_path, "a", encoding="utf-8") as file:
            file.write(request.content + "\n")

        return {"message": "Archivo guardado exitosamente", "path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/guardar_docs/")
async def guardar_archivo(request: ConsolidatedFileRequest):
    try:
        session_id = request.session_id
        os.makedirs("archivos_consolidados", exist_ok=True)
        filename = f"consolidado_{session_id}.txt"
        save_path = os.path.join("archivos_consolidados", filename)

        with open(save_path, "a", encoding="utf-8") as file:
            file.write(request.content + "\n")

        return {"message": "Archivo guardado exitosamente", "path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login/")
def login(usuario: Usuario):
    usuarios_validos = {
        "omar": "1234",
        "elsa": "abcd",
        "admin": "admin",
        "admin1": "admin"
    }

    if usuarios_validos.get(usuario.username) == usuario.password:
        session_id = f"{usuario.username}--{str(uuid.uuid4())}"
        return {"session_id": session_id, "username": usuario.username}
    else:
        raise HTTPException(status_code=401, detail="Usuario o contraseña inválidos")

@app.post("/validar/")
def ajustar(data: ValidacionInput):
    try:
        session_data = load_session_data(data.session_id)
        data_loaded = session_data.get("data", [])
        nuevos_codigos = data.nuevos_codigos
        feedback = data.feedback

        answer = call_azure_openai(
            f"La data es la siguiente ***{str(data_loaded)}*** los codigos que el usuario establecio ***{str(nuevos_codigos)}*** el feedback que dio el usuario es {feedback}",
            1,
            temperature=data.temperature,
            max_tokens=data.max_tokens,
            top_p=data.top_p,
            frequency_penalty=data.frequency_penalty,
            presence_penalty=data.presence_penalty
        )

        save_session_data(data.session_id, "codes", nuevos_codigos)
        save_session_data(data.session_id, "feedback", feedback)

        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class JustificacionInput(BaseModel):
    codigos: List[str]
    data: str

@app.post("/justificacion/")
def justificar_codigos(input: JustificacionInput):
    try:
        state = {
            "data": [input.data],
            "codes": [",".join(input.codigos)],
            "questions": [],
            "annotations": [],
        }
        result = validate_data(state)
        anotaciones = result["annotations"][0]
        return {"anotaciones": anotaciones}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

prompt_etiquetado = {
    "role": "system", 
    "content": "Eres un experto codificador de análisis cualitativo. Tu tarea es analizar párrafos de texto y asignarles códigos relevantes de una lista predefinida. Para cada párrafo, identifica qué códigos de la lista proporcionada son más relevantes. Un párrafo puede tener múltiples códigos si aplica. Responde únicamente con un JSON válido en este formato exacto: [{\"texto\": \"texto del párrafo\", \"codigos\": [\"codigo1\", \"codigo2\"]}, ...]"
}

class EtiquetadoRequest(BaseModel):
    session_id: str
    codigos: List[str]
    pagina: int = 1
    por_pagina: int = 20
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

@app.post("/etiquetar/")
def etiquetar_texto(request: EtiquetadoRequest):
    try:
        session_id = request.session_id
        codigos_aprobados = request.codigos
        pagina = max(1, request.pagina)
        por_pagina = min(max(5, request.por_pagina), 50)
        
        if not session_id or not codigos_aprobados:
            raise HTTPException(status_code=400, detail="session_id y codigos son requeridos")
        
        if len(codigos_aprobados) == 0:
            raise HTTPException(status_code=400, detail="Debe proporcionar al menos un código aprobado")
        
        file_path = f"./archivos_consolidados/consolidado_{session_id}.txt"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Archivo consolidado no encontrado. Complete los pasos 1 y 2 primero.")
            
        with open(file_path, "r", encoding="utf-8") as f:
            texto_completo = f.read()
        
        if not texto_completo.strip():
            raise HTTPException(status_code=400, detail="El archivo consolidado está vacío")
        
        session_data = load_session_data(session_id)
        segmentos_existentes = session_data.get("segmentos_codificados", [])
        
        if not segmentos_existentes:
            print("Procesando texto completo por primera vez...")
            segmentos_codificados = procesar_todo_el_texto(texto_completo, codigos_aprobados, request.temperature, request.max_tokens, request.top_p, request.frequency_penalty, request.presence_penalty)
            save_session_data(session_id, "segmentos_codificados", segmentos_codificados)
            save_session_data(session_id, "codigos_usados", list(set(c for s in segmentos_codificados for c in s["codigos"])))
        else:
            segmentos_codificados = segmentos_existentes
        
        total_segmentos = len(segmentos_codificados)
        inicio = (pagina - 1) * por_pagina
        fin = inicio + por_pagina
        
        segmentos_pagina = segmentos_codificados[inicio:fin]
        total_paginas = (total_segmentos + por_pagina - 1) // por_pagina
        
        codigos_usados = list(set(c for s in segmentos_codificados for c in s["codigos"]))
        
        print(f"Página {pagina}/{total_paginas}: {len(segmentos_pagina)} segmentos")
        
        return {
            "segmentos": segmentos_pagina,
            "paginacion": {
                "pagina_actual": pagina,
                "total_paginas": total_paginas,
                "total_segmentos": total_segmentos,
                "por_pagina": por_pagina,
                "tiene_siguiente": pagina < total_paginas,
                "tiene_anterior": pagina > 1
            },
            "codigos_usados": codigos_usados,
            "estadisticas": {
                "total_segmentos": total_segmentos,
                "codigos_aplicados": len(codigos_usados),
                "promedio_codigos_por_segmento": round(sum(len(s["codigos"]) for s in segmentos_codificados) / max(total_segmentos, 1), 2)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error general en etiquetado: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")
    
@app.post("/familias/")
def generar_familias(request: FamiliasRequest):
    try:
        codigos_aprobados = [c.strip() for c in request.codigos_aprobados if c.strip()]
        if not codigos_aprobados:
            raise HTTPException(status_code=400, detail="Debe proporcionar al menos un código aprobado.")

        # Construimos el prompt para el LLM
        lista_codigos_str = "\n".join(f"- {c}" for c in codigos_aprobados)
        prompt_usuario = f"""
        Tienes la siguiente lista de códigos de análisis cualitativo:

        {lista_codigos_str}

        Agrúpalos en familias temáticas siguiendo estrictamente estas reglas:
        1. Usa SOLO los códigos de la lista anterior.
        2. Cada código debe pertenecer a UNA SOLA familia (sin solapamientos).
        3. El nombre de cada familia puede ser una palabra o frase NUEVA que generalice los códigos.
        4. Devuelve la respuesta en JSON válido, en el siguiente formato:

        [
          {{
            "familia": "nombre_de_la_familia",
            "descripcion": "breve descripción de la familia",
            "codigos": ["codigo1", "codigo2"]
          }}
        ]
        """

        respuesta = call_azure_openai(
            prompt_usuario,
            4,  # opción para familias
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty
        )

        # Intentar parsear el JSON devuelto
        try:
            familias = json.loads(respuesta)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail=f"El modelo no devolvió un JSON válido: {respuesta[:200]}"
            )

        if not isinstance(familias, list):
            raise HTTPException(status_code=500, detail="El JSON de respuesta debe ser una lista de familias.")

        # Post-procesamiento: asegurar que ningún código pertenezca a más de una familia
        codigos_set = set(codigos_aprobados)
        codigos_asignados = set()
        familias_limpias = []

        for fam in familias:
            codigos_fam = fam.get("codigos", [])
            if not isinstance(codigos_fam, list):
                codigos_fam = []

            codigos_unicos = []
            for c in codigos_fam:
                # Solo conservar códigos válidos y que no hayan sido asignados
                if c in codigos_set and c not in codigos_asignados:
                    codigos_unicos.append(c)
                    codigos_asignados.add(c)

            if codigos_unicos:
                fam["codigos"] = codigos_unicos
                # Nombre y descripción por si el modelo no los retorna bien
                fam["familia"] = fam.get("familia", "Familia sin nombre")
                fam["descripcion"] = fam.get("descripcion", "")
                familias_limpias.append(fam)

        # Códigos que quedaron sin asignar → los ponemos en una familia "Otros"
        codigos_restantes = [c for c in codigos_aprobados if c not in codigos_asignados]
        if codigos_restantes:
            familias_limpias.append({
                "familia": "Otros códigos",
                "descripcion": "Códigos que no fueron agrupados claramente con otros.",
                "codigos": codigos_restantes
            })

        return {
            "familias": familias_limpias,
            "total_familias": len(familias_limpias),
            "total_codigos": len(codigos_aprobados)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando familias: {str(e)}")


def procesar_todo_el_texto(texto_completo: str, codigos_aprobados: List[str], temperature: float = 0.7, max_tokens: int = None, top_p: float = 1.0, frequency_penalty: float = 0.0, presence_penalty: float = 0.0) -> List[dict]:
    parrafos_raw = texto_completo.split('\n\n')
    parrafos = []
    
    for parrafo in parrafos_raw:
        parrafo = parrafo.strip()
        if parrafo and len(parrafo.split()) > 10:
            parrafos.append(parrafo)
    
    if len(parrafos) < 2:
        oraciones = [s.strip() for s in texto_completo.split('.') if s.strip() and len(s.split()) > 5]
        parrafos = []
        parrafo_actual = ""
        
        for oracion in oraciones:
            parrafo_actual += oracion + ". "
            if len(parrafo_actual.split()) > 15:
                parrafos.append(parrafo_actual.strip())
                parrafo_actual = ""
        
        if parrafo_actual.strip():
            parrafos.append(parrafo_actual.strip())
    
    print(f"Encontrados {len(parrafos)} párrafos para procesar")
    
    segmentos_codificados = []
    
    for i, parrafo in enumerate(parrafos[:25]):
        try:
            prompt_usuario = f"""
            Analiza este párrafo y asigna los códigos más relevantes de la lista proporcionada.
            
            PÁRRAFO: "{parrafo}"
            
            CÓDIGOS DISPONIBLES: {", ".join(codigos_aprobados)}
            
            INSTRUCCIONES:
            - Asigna 1-3 códigos máximo que sean más relevantes para este párrafo
            - Si ningún código aplica perfectamente, elige los más cercanos temáticamente
            - Considera el contexto completo del párrafo
            """
            
            respuesta = call_azure_openai(prompt_usuario, 3, temperature=temperature, max_tokens=max_tokens, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
            
            try:
                resultado = json.loads(respuesta)
                if isinstance(resultado, list) and len(resultado) > 0:
                    segmento = resultado[0]
                    segmento["codigos"] = [
                        c for c in segmento.get("codigos", []) 
                        if c in codigos_aprobados
                    ][:3]
                    
                    if segmento["codigos"]:
                        segmentos_codificados.append({
                            "texto": segmento.get("texto", parrafo),
                            "codigos": segmento["codigos"]
                        })
                        
            except json.JSONDecodeError:
                codigos_encontrados = []
                respuesta_lower = respuesta.lower()
                
                for codigo in codigos_aprobados:
                    if (codigo.lower() in respuesta_lower or 
                        any(word in respuesta_lower for word in codigo.lower().split())):
                        codigos_encontrados.append(codigo)
                
                if codigos_encontrados:
                    segmentos_codificados.append({
                        "texto": parrafo,
                        "codigos": codigos_encontrados[:3]
                    })
                    
        except Exception as e:
            print(f"Error procesando párrafo {i+1}: {e}")
            continue
    
    print(f"Procesamiento completado: {len(segmentos_codificados)} segmentos codificados")
    return segmentos_codificados

import csv
from io import StringIO
from fastapi.responses import StreamingResponse

@app.get("/exportar_csv/{session_id}")
def exportar_segmentos_csv(session_id: str):
    try:
        session_data = load_session_data(session_id)
        segmentos_codificados = session_data.get("segmentos_codificados", [])
        
        if not segmentos_codificados:
            raise HTTPException(status_code=404, detail="No hay segmentos codificados para exportar")
        
        output = StringIO()
        writer = csv.writer(output, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow([
            'id_segmento',
            'texto_segmento', 
            'codigos_aplicados',
            'numero_codigos',
            'longitud_texto'
        ])
        
        for i, segmento in enumerate(segmentos_codificados, 1):
            codigos_str = '|'.join(segmento.get('codigos', []))
            writer.writerow([
                i,
                segmento.get('texto', '').replace('\n', ' ').replace('\r', ''),
                codigos_str,
                len(segmento.get('codigos', [])),
                len(segmento.get('texto', ''))
            ])
        
        output.seek(0)
        response = StreamingResponse(
            io.StringIO(output.getvalue()),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=segmentos_codificados_{session_id}.csv"
            }
        )
        
        print(f"Exportado CSV con {len(segmentos_codificados)} segmentos para sesión {session_id}")
        return response
        
    except Exception as e:
        print(f"Error exportando CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error exportando CSV: {str(e)}")

@app.get("/historial/{session_id}")
def obtener_historial(session_id: str):
    try:
        session_path = f"sessions/session_{session_id}.json"
        if not os.path.exists(session_path):
            raise HTTPException(status_code=404, detail="No se encontraron datos para esta sesión.")

        with open(session_path, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        # CORRECTED: Read from archivos_consolidados instead of archivos_guardados
        consolidado_path = f"./archivos_consolidados/consolidado_{session_id}.txt"
        texto_consolidado = ""
        if os.path.exists(consolidado_path):
            with open(consolidado_path, "r", encoding="utf-8") as f:
                texto_consolidado = f.read()

        timestamp = datetime.fromtimestamp(os.path.getmtime(session_path)).isoformat()

        return JSONResponse(content={
            "session_id": session_id,
            "timestamp": timestamp,
            "preguntas": session_data.get("questions", []),
            "codigos": session_data.get("codes", []),
            "feedback": session_data.get("feedback", ""),
            "texto_consolidado": texto_consolidado
        }, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener historial: {str(e)}")































 