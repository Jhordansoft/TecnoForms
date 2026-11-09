
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image, ImageOps
from pathlib import Path

# --------------------------------
# CONFIGURACIÓN
# --------------------------------

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


# --------------------------------
# CARGAR MODELO
# --------------------------------

@st.cache_resource
def load_model():
    ruta = Path(__file__).parent / "modelo_digitos.keras"

    if not ruta.exists():
        raise FileNotFoundError("No se encontró el modelo entrenado.")

    return tf.keras.models.load_model(
        ruta,
        compile=False
    )


try:
    model = load_model()
    st.success("Red neuronal cargada correctamente.v2")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()


# --------------------------------
# UMBRAL AUTOMÁTICO (OTSU)
# --------------------------------

def calcular_umbral(image):

    pixels = np.asarray(image)

    histograma = np.bincount(
        pixels.flatten(),
        minlength=256
    )

    total = pixels.size
    suma_total = np.dot(np.arange(256), histograma)

    suma_fondo = 0
    peso_fondo = 0
    mejor_varianza = 0
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


# --------------------------------
# PREPROCESAMIENTO
# --------------------------------

def procesar_imagen(image):

    # Escala de grises
    image = ImageOps.grayscale(image)

    # Mejorar contraste
    image = ImageOps.autocontrast(image)

    # Detectar el color del fondo usando los bordes
    pixels = np.asarray(image)

    borde = np.concatenate([
        pixels[0, :],
        pixels[-1, :],
        pixels[:, 0],
        pixels[:, -1]
    ])

    # MNIST utiliza fondo negro y trazos blancos
    if np.median(borde) > 127:
        image = ImageOps.invert(image)

    # Calcular umbral automático
    umbral = calcular_umbral(image)

    # Crear máscara para detectar el dígito
    mask = image.point(
        lambda p: 255 if p > umbral else 0
    )

    bbox = mask.getbbox()

    if bbox is None:
        raise ValueError("No se pudo detectar ningún dígito.")

    # Recortar el número
    digit = image.crop(bbox)

    width, height = digit.size

    # Conservar proporciones y ajustar a 20x20
    scale = min(20 / width, 20 / height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    digit = digit.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Lienzo negro de 28x28
    canvas = Image.new("L", (28, 28), 0)

    # Centrar el número
    x = (28 - new_width) // 2
    y = (28 - new_height) // 2

    canvas.paste(digit, (x, y))

    # Normalización
    array = np.asarray(canvas).astype("float32") / 255.0

    # Dimensiones para CNN
    array = np.expand_dims(array, axis=(0, -1))

    return canvas, array


# --------------------------------
# SUBIR IMAGEN
# --------------------------------

st.divider()

archivo = st.file_uploader(
    "Selecciona una imagen con un solo dígito",
    type=["png", "jpg", "jpeg"]
)

if archivo is not None:

    try:

        original = Image.open(archivo).convert("RGB")

        if original.width * original.height > 12000000:
            st.error("La imagen es demasiado grande.")
            st.stop()

        procesada, entrada = procesar_imagen(original)
        st.write("### Diagnóstico de la imagen")

st.write("Dimensiones:", procesada.size)
st.write("Valor mínimo:", float(np.min(entrada)))
st.write("Valor máximo:", float(np.max(entrada)))
st.write("Promedio:", float(np.mean(entrada)))
        # Mostrar imágenes
        st.markdown("### Procesamiento de imagen")

        col1, col2 = st.columns(2)


        with col1:
            st.write("Imagen original")
            st.image(
                original,
                caption="Imagen subida",
                output_format="PNG",
                use_container_width=True
            )
        
        with col2:
            st.write("Imagen procesada (28x28)")
            st.image(
                procesada.resize(
                    (280, 280),
                    Image.Resampling.NEAREST
                ),
                caption="Entrada de la red neuronal",
                output_format="PNG",
                use_container_width=True
            )


        # --------------------------------
        # PREDICCIÓN
        # --------------------------------

        resultado = model.predict(
            entrada,
            verbose=0
        )[0]

        numero = int(np.argmax(resultado))
        confianza = float(resultado[numero]) * 100

        st.divider()

        st.markdown("### 🧠 Resultado del reconocimiento")

        st.metric(
            label="Dígito identificado",
            value=str(numero),
            delta=f"{confianza:.2f}% de confianza"
        )

        if confianza >= 80:
            st.success("Reconocimiento con confianza alta.")

        elif confianza >= 50:
            st.warning("Reconocimiento con confianza moderada.")

        else:
            st.error(
                "El modelo tiene dificultades para reconocer "
                "este número. Intenta con otra imagen."
            )

        # --------------------------------
        # PROBABILIDADES
        # --------------------------------

        st.markdown("### 📊 Probabilidad por dígito")

        st.bar_chart({
            str(i): float(resultado[i]) * 100
            for i in range(10)
        })

        # Mostrar las tres predicciones principales
        st.markdown("### Tres predicciones principales")

        mejores = np.argsort(resultado)[-3:][::-1]

        for i in mejores:
            st.write(
                f"Dígito **{i}**: "
                f"{resultado[i] * 100:.2f}%"
            )

    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")


st.divider()

st.caption(
    "TecnoForms | Clasificación mediante "
    "Redes Neuronales Convolucionales (CNN)"
)
