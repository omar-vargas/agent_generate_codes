
import streamlit as st
from streamlit_tags import st_tags
from streamlit_lottie import st_lottie
import requests
import os
import json
import pandas as pd
import ast
import re
from pathlib import Path

import uuid


# Inicializar el paso
if "step" not in st.session_state:
    st.session_state.step = 1

# Funciones de navegación
def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1

# Cargar animación
def load_lottiefile(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

lottie_thinking = load_lottiefile("Animation.json") 

st.set_page_config(page_title="Evaluación cualitativa", page_icon="🧠")


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

import uuid
import requests
import streamlit as st

# Inicializar estado de sesión
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Función para iniciar sesión
def login(username, password):
    try:
        response = requests.post("http://localhost:8000/login/", json={
            "username": username,
            "password": password
        })
        if response.status_code == 200:
            data = response.json()
            st.session_state.logged_in = True
            st.session_state.username = data["username"]
            st.session_state.session_id = data["session_id"]
            st.rerun()  # ← Esto es lo que fuerza avanzar a la siguiente página

        else:
            st.error("❌ Usuario o contraseña incorrectos.")
    except Exception as e:
        st.error(f"Error de conexión: {e}")

if not st.session_state.logged_in:
    st.title("🔐 Inicio de sesión")
    with st.form("login_form"):
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Iniciar sesión")
        if submitted:
            login(username, password)
    st.stop()  # Detiene la ejecución hasta que el usuario inicie sesión


st.title("🧪 Evaluación Cualitativa Asistida por Agentes")
st.sidebar.markdown(f"🆔 Session ID: `{st.session_state.session_id}`")

# Paso 1: Subir archivos y generar códigos
if st.session_state.step == 1:
    st.header("Paso 1: Subir archivos y generar códigos")

    uploaded_files = st.file_uploader("Sube archivos .txt", type="txt", accept_multiple_files=True)
    


# Ruta base: carpeta del script actual
    BASE_DIR = Path(__file__).resolve().parent

# Ruta al directorio "back" desde "front"
    back_path = BASE_DIR.parent / "backend"
    session_id = st.session_state.session_id

    ruta_destino = back_path / f"consolidado_{session_id}.txt"




    def consolidar_archivos(files, ruta_destino):
        contenido_total = []
        for idx, file in enumerate(files):
            texto = file.getvalue().decode("utf-8")
            delimitador = f"\n\n-- Inicio del archivo {idx + 1} --\n\n"
            contenido_total.append(delimitador)
            contenido_total.append(texto)
            contenido_total.append(f"\n\n-- Fin del archivo {idx + 1} --\n\n")
        contenido_consolidado = ''.join(contenido_total)
        with open(ruta_destino, "w", encoding="utf-8") as f:
            f.write(contenido_consolidado)
        return contenido_consolidado

    if uploaded_files:
        contenido_consolidado = consolidar_archivos(uploaded_files, ruta_destino)
        st.success("✅ Archivos consolidados correctamente.")
        keywords = st_tags(label='🎯 Hipótesis u objetivos de investigación',
                           text='Presiona enter para añadir',
                           value=[], maxtags=10, key='step1_tags')

        if st.button("➡️ Siguiente"):
            st.session_state.keywords = keywords
            next_step()
            st.rerun()  # ← importante


# Paso 2: Revisión del agente
elif st.session_state.step == 2:
    st.header("Paso 2: Revisión de códigos por el agente")

    if "keywords" in st.session_state:
        data = {"preguntas": str(st.session_state.keywords),
                   "session_id": st.session_state.session_id }
        with st.spinner("Generando códigos..."):
            st_lottie(lottie_thinking, height=200)
            response = requests.post("http://localhost:8000/generar/",
                                     headers={"Content-Type": "application/json"},
                                     data=json.dumps(data))
        if response.status_code == 200:
            st.success("✅ Códigos generados correctamente.")
            codigos_str = response.json()[0]
            codigos_limpios = ast.literal_eval(codigos_str)
            lista_codigos = [codigo.strip() for codigo in codigos_limpios[0].split(",") if codigo.strip()]
            df = pd.DataFrame(lista_codigos, columns=["Códigos"])
            st.session_state.codes = df
            st.dataframe(df, use_container_width=True)
        else:
            st.error("Error al obtener códigos")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.button("⬅️ Anterior", on_click=prev_step)
        with col2:
            st.button("➡️ Siguiente", on_click=next_step)
    else:
        st.warning("Por favor completa el paso anterior.")
        if st.button("⬅️ Volver"):
            prev_step()

elif st.session_state.step == 3:
    st.header("Paso 3: Feedback y ajustes finales")

    if "codes" in st.session_state:
        if "step3_tags" not in st.session_state:
            st.session_state["step3_tags"] = st.session_state.codes["Códigos"].tolist()

        if "tag_input_key" not in st.session_state:
            st.session_state["tag_input_key"] = 0

        st.subheader("✍️ Edita los códigos sugeridos")

# Campo editable con todos los tags acumulados
        keywords_codes = st_tags(
            label='🗂️ Edita los códigos',
            text='Presiona Enter para agregar más',
            value=st.session_state["step3_tags"],
            maxtags=50,
            key=f'step3_tags_{st.session_state["tag_input_key"]}'
        )

        # Vista diferenciada por iteración justo debajo
        st.markdown("**🎨 Códigos agrupados por feedback:**")

        color_map = ["#00cc66", "#3399ff", "#ff9933", "#cc33cc", "#ff6666"]
        tag_history = st.session_state.get("tag_history", {})

        for idx, (etiqueta, tags) in enumerate(tag_history.items()):
            color = color_map[idx % len(color_map)]
            st.markdown(f"<b>{etiqueta}</b>", unsafe_allow_html=True)
            tag_line = " ".join([
                f"<span style='background-color:{color}; padding:6px 10px; border-radius:20px; color:white; margin:4px; display:inline-block'>{tag}</span>"
                for tag in tags
            ])
            st.markdown(tag_line, unsafe_allow_html=True)


        st.markdown("**💡 Visualización por iteración:**")

        color_map = ["#00cc66", "#3399ff", "#ff9933", "#cc33cc", "#ff6666"]
        tag_history = st.session_state.get("tag_history", {})

        if tag_history:
            for idx, (etiqueta, tags) in enumerate(tag_history.items()):
                color = color_map[idx % len(color_map)]
                st.markdown(f"<b>{etiqueta}</b>", unsafe_allow_html=True)
                tag_line = " ".join([
                    f"<span style='background-color:{color}; padding:6px 10px; border-radius:20px; color:white; margin:4px; display:inline-block'>{tag}</span>"
                    for tag in tags
                ])
                st.markdown(tag_line, unsafe_allow_html=True)


        feedback = st.text_area("💬 ¿Qué opinas de los códigos generados?",
                                placeholder="Ejemplo: Falta énfasis en temas emocionales...")

        aprobado = st.checkbox("✅ Estoy satisfecho con los códigos generados")

        if "feedback_list" not in st.session_state:
            st.session_state["feedback_list"] = []

        if st.button("📤 Enviar feedback al agente"):
            st.session_state["feedback_list"].append(feedback)
            payload = {
                "aprobado": aprobado,
                "nuevos_codigos": keywords_codes,
                "feedback": feedback,
                "session_id": st.session_state.session_id
            }
            response = requests.post("http://localhost:8000/validar/", json=payload)
            if response.status_code == 200:
                st.success("✅ Feedback enviado con éxito.")
                new_codigos_str = response.json()
                try:
                    new_codigos = [codigo.strip() for codigo in new_codigos_str.split(",") if codigo.strip()]
                    iteracion = st.session_state.get("feedback_round", 1)

                    # Guardamos los códigos por iteración
                    if "tag_history" not in st.session_state:
                        st.session_state["tag_history"] = {}

                    st.session_state["tag_history"][f"Iteración {iteracion}"] = new_codigos
                    st.session_state["feedback_round"] = iteracion + 1

                    # Acumulamos todos
                    updated_tags = list(set(st.session_state["step3_tags"] + new_codigos))
                    st.session_state["step3_tags"] = updated_tags

                    st.session_state["tag_input_key"] += 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al procesar la respuesta: {e}")


        # Solo mostrar el botón de pasar al resumen si el usuario aprueba
        if aprobado:
            st.button("✅ Estoy conforme, ver resumen final", on_click=next_step)


        st.button("⬅️ Anterior", on_click=prev_step)

elif st.session_state.step == 4:
    st.header("📊 Paso 4: Resumen Final")

    # Mostrar códigos
    st.subheader("🗂️ Códigos Finales")
    if "step3_tags" in st.session_state:
        df_codigos = pd.DataFrame(st.session_state["step3_tags"], columns=["Códigos"])
        st.dataframe(df_codigos)

        csv_codigos = df_codigos.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar códigos en CSV", data=csv_codigos,
                           file_name="codigos_finales.csv", mime="text/csv")
        
        # Guardar en backend
        requests.post("http://localhost:8000/guardar/", json={"content": df_codigos.to_csv(index=False)})

    
    else:
        st.warning("No hay códigos finales disponibles.")


    # Mostrar preguntas
    st.subheader("📌 Preguntas de Investigación")
    if "keywords" in st.session_state:
        preguntas_texto = "\n".join(st.session_state["keywords"])
        st.text_area("Preguntas", value=preguntas_texto, height=150)
        st.download_button("⬇️ Descargar preguntas",
                        data=preguntas_texto, file_name="preguntas_investigacion.txt", mime="text/plain")
        requests.post("http://localhost:8000/guardar/", json={"content": preguntas_texto})

    else:
        st.warning("No hay preguntas registradas.")


    # Mostrar feedback
    st.subheader("💬 Feedback del Usuario")
    if "feedback_list" in st.session_state and st.session_state["feedback_list"]:
        feedback_texto = "\n\n".join(st.session_state["feedback_list"])
        st.text_area("Feedback acumulado", value=feedback_texto, height=200)
        st.download_button("⬇️ Descargar feedback",
                           data=feedback_texto, file_name="feedback_usuario.txt", mime="text/plain")
        requests.post("http://localhost:8000/guardar/", json={"content": feedback_texto})
    else:
        st.warning("No hay feedback acumulado.")

    # Botón para reiniciar
    st.subheader("🔁 ¿Deseas iniciar una nueva evaluación?")
    if st.button("🔄 Reiniciar aplicación"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    else:
        st.warning("No hay códigos generados. Regresa al paso 2.")
        if st.button("⬅️ Volver"):
            prev_step()
