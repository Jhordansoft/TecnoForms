
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image, ImageOps
from pathlib import Path

# CONFIGURACIÓN
st.set_page_config(
    page_title="TecnoForms",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 TecnoForms")
st.subheader("Clasificación de Dígitos Manuscritos")

st.write(
    "Sube una imagen de un número escrito a mano "
    "y nuestra inteligencia artificial lo reconocerá."
)

# CARGAR MODELO
@st.cache_resource
def load_model():
    ruta = Path(__file__).parent / "modelo_digitos.keras"
    return tf.keras.models.load_model(ruta, compile=False)

try:
    model = load_model()
    st.success("Red neuronal cargada correctamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()


# PROCESAMIENTO DE IMAGEN
def procesar_imagen(image):

    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)

    # Invertir colores si el fondo es blanco
    if np.mean(image) > 127:
        image = ImageOps.invert(image)

    # Detectar el número
    mask = image.point(lambda p: 255 if p > 35 else 0)
    bbox = mask.getbbox()

    if bbox is None:
        raise ValueError("No se detectó ningún dígito.")

    image = image.crop(bbox)

    # Ajustar conservando proporciones
    width, height = image.size
    scale = min(20 / width, 20 / height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Crear lienzo de 28x28
    canvas = Image.new("L", (28, 28), 0)

    x = (28 - new_width) // 2
    y = (28 - new_height) // 2

    canvas.paste(image, (x, y))

    array = np.asarray(canvas).astype("float32") / 255.0
    array = np.expand_dims(array, axis=(0, -1))

    return canvas, array


# SUBIR IMAGEN
st.divider()

archivo = st.file_uploader(
    "Selecciona una imagen con un solo número",
    type=["png", "jpg", "jpeg"]
)

if archivo is not None:

    try:
        original = Image.open(archivo).convert("RGB")

        if original.width * original.height > 12000000:
            st.error("La imagen es demasiado grande.")
            st.stop()

        procesada, entrada = procesar_imagen(original)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Imagen original**")
            st.image(original, use_container_width=True)

        with col2:
            st.markdown("**Imagen procesada**")
            st.image(
                procesada.resize((280, 280)),
                use_container_width=True
            )

        # PREDICCIÓN
        resultado = model.predict(entrada, verbose=0)[0]

        numero = int(np.argmax(resultado))
        confianza = float(resultado[numero]) * 100

        st.divider()

        st.markdown("### Resultado del reconocimiento")

        st.metric(
            "Dígito identificado",
            numero,
            f"{confianza:.2f}% de confianza"
        )

        if confianza >= 70:
            st.success("¡Dígito reconocido!")
        else:
            st.warning("Confianza baja. Intenta con otra imagen.")

        st.markdown("### Probabilidad por dígito")

        st.bar_chart({
            str(i): float(resultado[i])
            for i in range(10)
        })

    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")

st.divider()

st.caption("TecnoForms | Machine Learning y Deep Learning")
