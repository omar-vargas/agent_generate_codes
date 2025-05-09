from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import List
from typing import TypedDict,NotRequired
from langgraph.graph import StateGraph, START, END
import os

import random
from typing import Literal
import ast
import re
import json

from datetime import datetime
import uuid

load_dotenv()



os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")




print(" Variables de entorno cargadas correctamente.")



from typing import Annotated
import operator
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

prompt_agente_generador_codigos= {
            "role": "system",
            "content": "Eres un experto generador de códigos de evaluación cualitativa. Tu tarea consiste en analizar textos de diversas fuentes de datos y generar códigos que correspondan a las ideas o conceptos clave que surjan en los textos. El usuario te proporcionará los objetivos o hipotesis de la investigacion cualitativa  y la data de la investigacion de entrevistas para que los utilices para generar estos códigos. como respuesta debe dar solo la lista de codigos separadada por comas nada mas no añadas comentarios ni antes ni despues"
        }


prompt_agente_generador_codigos_feedback= {
            "role": "system",
            "content": "Eres un experto generador de códigos de evaluación cualitativa. Tu tarea consiste en analizar textos de diversas fuentes de datos y generar códigos que correspondan a las ideas o conceptos clave que surjan en los textos. El usuario te proporcionará los codigos generados hasta ahora ,la data de la investigacion de entrevistas y por ultimo un feedback para que lo tengas en cuenta a la hora de generar nuevos codigos para que los utilices . como respuesta debe dar solo la lista de codigos separadada por comas nada mas no añadas comentarios ni antes ni despues"
        }


prompt_agente_corrector_codigos= {
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




# Definir el tipo de estado
# Definir el tipo de estado usando 'Annotated' para permitir múltiples valores de 'questions'
class State(TypedDict):
    data: Annotated[list[str], operator.add]  # Usamos lista de cadenas
    questions: Annotated[list[str], operator.add]
    question: Annotated[list[str], operator.add]    # Usamos lista para permitir múltiples valores de 'questions'
    codes: Annotated[list[str], operator.add]
    final_codes:  Annotated[list[str], operator.add]
    annotations:  Annotated[list[str], operator.add]
    validate:  bool
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://invuniandesai-2.openai.azure.com/"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY", "DEFAULT_KEY"),
    api_version="2024-10-21"
)

def run_prompt(client, prompt_message: str,role_agent:dict, model: str = "gpt", tokens=0 ):
    # Realizar la solicitud al modelo de OpenAI
    response = client.chat.completions.create(

        model=model,
        messages=[ role_agent,  {"role": "user", "content": prompt_message}]
    )
    
    # Imprimir la respuesta completa para inspeccionar su estructura
    
    
    # Obtener el número total de tokens usados
    tokens += response.usage.total_tokens
    
    # Intentar acceder al contenido de manera más segura
    content = response.choices[0].message.content  # Esto depende de la estructura

    return content, tokens

def call_azure_openai(prompt: str, option:int) -> str:
    try:
       
        if option==0:
            answer, tokens_used = run_prompt(client, prompt, prompt_agente_generador_codigos)
        elif option==1:
             answer, tokens_used = run_prompt(client, prompt, prompt_agente_generador_codigos_feedback)
        else: 
            answer, tokens_used = run_prompt(client, prompt, prompt_agente_corrector_codigos)
        return answer
    except Exception as e:
        return f"Error al procesar la solicitud: {str(e)}"


# Nodo 1 (modificando 'questions' de forma acumulativa)
def node_1(state: State) -> State:
    if "data" not in state:
        state["data"] = []  # Inicializamos la lista si no existe
    if "questions" not in state:
        state["questions"] = []  # Inicializamos 'questions' como lista si no existe
    answer = call_azure_openai('La data es la siguiente ***'+str(state["data"])+'***'+' las preguntas son ***'+str(state["questions"]),0)

    state["codes"]=[answer]

    return state




def validate_data(state: State) -> State:
    if "data" not in state:
        state["data"] = []
    if "questions" not in state:
        state["questions"] = []
    if "annotations" not in state:
        state["annotations"] = []

    answer = call_azure_openai(
        f'La data es la siguiente ***{str(state["data"])}*** los codigos generados fueron ***{str(state["codes"])}***',
        2
    )

    state["annotations"] = [answer]

    # Guardamos temporalmente el estado en result_storage
    result_storage["estado_actual"] = state

    return state


def human_aprobation(state: State, approved: bool = True, feedback:str ='muchos codigos') -> Literal['node_1',END]:

    is_approved = interrupt(
        True
        
    )


    if state['validate']==True:
        return END       
    else:
        state['codes']
        return node_1
       
    


# Crear el grafo de estados
builder = StateGraph(State)

# Añadir nodos al grafo
builder.add_node('node_1', node_1)
builder.add_node('validate_data', validate_data)

# Añadir transiciones entre los nodos
builder.add_edge(START, 'node_1')

builder.add_edge('node_1', 'validate_data')  # Nodo intermedio para combinar
builder.add_conditional_edges('validate_data', human_aprobation)

  
  
# Ejecutar el grafo con el estado inicial
initial_state = {
    "data": ["percepcion de los estudiantes sobre la IA generatica"],  # Iniciar 'data' como una lista vacía
    "questions": ['los estudiantes cada ven pensaran menos y seran mas flojos']  # Iniciar 'questions' como una lista vacía
}


# Configurar FastAPI
app = FastAPI()

# Permitir solicitudes CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Reemplaza "*" por una lista de orígenes específicos si es necesario
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

class ValidacionInput(BaseModel):
    aprobado: bool
    nuevos_codigos: List[str] = []
    feedback: str
    session_id: str 

class ConsolidatedFileRequest(BaseModel):
    content: str
    session_id: str  


class Usuario(BaseModel):
    username: str
    password: str  




def load_data_from_txt(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")
    
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read().strip() 
    
    initial_state["data"] = [content]  



# Función para ejecutar el grafo
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


# Endpoints de FastAPI

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

        load_data_from_txt(f"./consolidado_{session_id}.txt")
        initial_state["questions"] = [preguntas]
        initial_state["validate"] = False

        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)
        result = graph.invoke(initial_state, {"configurable": {"thread_id": session_id}})

        save_session_data(session_id, "codes", result["codes"])
        save_session_data(session_id, "data", initial_state["data"])
        save_session_data(session_id, "questions", initial_state["questions"])

        return {str(result["codes"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



    
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


            

            return  temas,justificaciones
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

@app.post("/login/")
def login(usuario: Usuario):
    # Simulación de base de datos de usuarios
    usuarios_validos = {
        "omar": "1234",
        "elsa": "abcd"
    }

    if usuarios_validos.get(usuario.username) == usuario.password:
        # Reutilizamos session_id como token para ahora
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
            1
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

@app.get("/historial/{session_id}")
def obtener_historial(session_id: str):
    try:

        session_path = f"sessions/session_{session_id}.json"
        if not os.path.exists(session_path):
            raise HTTPException(status_code=404, detail="No se encontraron datos para esta sesión.")

        with open(session_path, "r", encoding="utf-8") as f:
            session_data = json.load(f)


        consolidado_path = f"./archivos_guardados/output_{session_id}.txt"
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
