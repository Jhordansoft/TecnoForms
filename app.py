
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image, ImageOps
from pathlib import Path


# ==========================================
# 1. CONFIGURACION
# ==========================================

st.set_page_config(
    page_title="TecnoForms",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 TecnoForms")
st.subheader("Clasificación de Dígitos Manuscritos")

st.write(
    "Sube una imagen con un único número escrito a mano "
    "y nuestra inteligencia artificial lo reconocerá."
)


# ==========================================
# 2. CARGAR MODELO
# ==========================================

@st.cache_resource
def load_model():

    ruta = Path(__file__).parent / "modelo_digitos.keras"

    if not ruta.exists():
        raise FileNotFoundError(
            "No se encontró modelo_digitos.keras"
        )

    return tf.keras.models.load_model(
        ruta,
        compile=False
    )


try:
    model = load_model()
    st.success("Red neuronal cargada correctamente.v5")

except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()


# ==========================================
# 3. CALCULAR UMBRAL OTSU
# ==========================================

def calcular_umbral(image):

    pixels = np.asarray(image)

    histograma = np.bincount(
        pixels.ravel(),
        minlength=256
    )

    total = pixels.size
    suma_total = np.dot(
        np.arange(256),
        histograma
    )

    suma_fondo = 0
    peso_fondo = 0
    mejor_varianza = -1
    umbral = 127

    for i in range(256):

        peso_fondo += histograma[i]

        if peso_fondo == 0:
            continue

        peso_objeto = total - peso_fondo

        if peso_objeto == 0:
            break

        suma_fondo += i * histograma[i]

        media_fondo = suma_fondo / peso_fondo

        media_objeto = (
            suma_total - suma_fondo
        ) / peso_objeto

        varianza = (
            peso_fondo * peso_objeto *
            (media_fondo - media_objeto) ** 2
        )

        if varianza > mejor_varianza:
            mejor_varianza = varianza
            umbral = i

    return umbral


# ==========================================
# 4. PREPROCESAMIENTO DE IMAGEN
# ==========================================


def procesar_imagen(image):

    # 1. Convertir a escala de grises
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)

    pixels = np.asarray(image)

    # 2. Detectar fondo mediante los bordes
    borde = np.concatenate([
        pixels[0, :],
        pixels[-1, :],
        pixels[:, 0],
        pixels[:, -1]
    ])

    # Convertir a fondo negro y trazos blancos
    if np.median(borde) > 127:
        image = ImageOps.invert(image)

    # 3. Umbral Otsu
    umbral = calcular_umbral(image)

    # 4. Crear máscara
    mask = image.point(
        lambda p: 255 if p > umbral else 0
    )

    bbox = mask.getbbox()

    if bbox is None:
        raise ValueError("No se detectó ningún dígito.")

    # 5. Recortar el número
    digit = image.crop(bbox)

    # 6. Ajustar a 20x20 sin deformar
    width, height = digit.size

    scale = min(20 / width, 20 / height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    digit = digit.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # 7. Crear lienzo de 28x28
    canvas = Image.new("L", (28, 28), 0)

    x = (28 - new_width) // 2
    y = (28 - new_height) // 2

    canvas.paste(digit, (x, y))

    # 8. Centrar según el centro de masa
    array = np.asarray(canvas).astype("float32")

    total = np.sum(array)

    if total > 0:

        yy, xx = np.indices(array.shape)

        centro_x = np.sum(xx * array) / total
        centro_y = np.sum(yy * array) / total

        desplazamiento_x = round(13.5 - centro_x)
        desplazamiento_y = round(13.5 - centro_y)

        canvas = canvas.transform(
            (28, 28),
            Image.Transform.AFFINE,
            (
                1, 0, -desplazamiento_x,
                0, 1, -desplazamiento_y
            ),
            resample=Image.Resampling.BILINEAR
        )

    # 9. Normalizar
    array = np.asarray(canvas).astype("float32") / 255.0

    # 10. Preparar entrada CNN
    entrada = np.expand_dims(array, axis=(0, -1))

    return canvas, entrada

# ==========================================
# 5. SUBIR IMAGEN
# ==========================================

st.divider()

st.markdown("### 📤 Subir imagen")

archivo = st.file_uploader(
    "Selecciona una imagen con un solo dígito",
    type=["png", "jpg", "jpeg"]
)


# ==========================================
# 6. PROCESAR Y RECONOCER
# ==========================================

if archivo is not None:

    try:

        # Abrir imagen
        original = Image.open(archivo).convert("RGB")

        # Comprobar tamaño
        if original.width * original.height > 12000000:
            st.error("La imagen es demasiado grande.")
            st.stop()

        # Procesar imagen
        procesada, entrada = procesar_imagen(original)

       
# ----------------------------------
# MOSTRAR IMAGENES
# ----------------------------------

import io

st.divider()
st.markdown("### 🖼️ Procesamiento de imagen")

col1, col2 = st.columns(2)

# Convertir la imagen original a PNG en memoria
buffer_original = io.BytesIO()
original.save(buffer_original, format="PNG")

# Convertir la imagen procesada a PNG en memoria
buffer_procesada = io.BytesIO()
procesada.resize(
    (280, 280),
    Image.Resampling.NEAREST
).save(buffer_procesada, format="PNG")

with col1:
    st.write("**Imagen original**")
    st.image(
        buffer_original.getvalue(),
        caption="Imagen subida",
        width="stretch"
    )

with col2:
    st.write("**Imagen procesada (28x28)**")
    st.image(
        buffer_procesada.getvalue(),
        caption="Entrada de la red neuronal",
        width="stretch"
    )

        # ----------------------------------
        # DIAGNOSTICO
        # ----------------------------------

        with st.expander("Ver diagnóstico de la imagen"):

            st.write("Dimensiones:", procesada.size)

            st.write(
                "Valor mínimo:",
                float(np.min(entrada))
            )

            st.write(
                "Valor máximo:",
                float(np.max(entrada))
            )

            st.write(
                "Promedio:",
                float(np.mean(entrada))
            )

            st.write(
                "Forma de entrada:",
                entrada.shape
            )

        # ----------------------------------
        # PREDICCION
        # ----------------------------------

        resultado = model.predict(
            entrada,
            verbose=0
        )[0]

        numero = int(np.argmax(resultado))

        confianza = float(
            resultado[numero]
        ) * 100

        # ----------------------------------
        # MOSTRAR RESULTADO
        # ----------------------------------

        st.divider()

        st.markdown(
            "### 🧠 Resultado del reconocimiento"
        )

        st.metric(
            label="Dígito identificado",
            value=str(numero),
            delta=f"{confianza:.2f}% de confianza"
        )

        if confianza >= 80:

            st.success(
                "Reconocimiento con confianza alta."
            )

        elif confianza >= 50:

            st.warning(
                "Reconocimiento con confianza moderada."
            )

        else:

            st.error(
                "El modelo tiene dificultades para "
                "reconocer este número."
            )

        # ----------------------------------
        # GRAFICO DE PROBABILIDADES
        # ----------------------------------

        st.markdown(
            "### 📊 Probabilidad por dígito"
        )

        probabilidades = {
            str(i): float(resultado[i]) * 100
            for i in range(10)
        }

        st.bar_chart(probabilidades)

        # ----------------------------------
        # TRES MEJORES PREDICCIONES
        # ----------------------------------

        st.markdown(
            "### 🔍 Tres predicciones principales"
        )

        mejores = np.argsort(resultado)[-3:][::-1]

        for i in mejores:

            st.write(
                f"**Dígito {i}:** "
                f"{resultado[i] * 100:.2f}%"
            )

    except Exception as e:

        st.error(
            f"Error al procesar la imagen: {e}"
        )


# ==========================================
# 7. PIE DE PAGINA
# ==========================================

st.divider()

st.caption(
    "TecnoForms | Sistema de clasificación "
    "mediante Redes Neuronales Convolucionales (CNN)"
)
