
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image, ImageOps
from pathlib import Path
import io


# =====================================================
# 1. CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="TecnoForms",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 TecnoForms")
st.subheader("Clasificación de Dígitos Manuscritos")

st.write(
    "Sube una imagen que contenga un único número escrito "
    "a mano y nuestra red neuronal intentará reconocerlo."
)


# =====================================================
# 2. CARGAR MODELO
# =====================================================

@st.cache_resource
def load_model():

    ruta = Path(__file__).parent / "modelo_digitos.keras"

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {ruta}"
        )

    modelo = tf.keras.models.load_model(
        ruta,
        compile=False
    )

    return modelo


try:
    model = load_model()
    st.success("✅ Red neuronal cargada correctamente.")

except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()


# =====================================================
# 3. UMBRALIZACIÓN DE OT SU
# =====================================================

def calcular_umbral_otsu(imagen):

    pixels = np.asarray(imagen, dtype=np.uint8)

    histograma = np.bincount(
        pixels.ravel(),
        minlength=256
    )

    total = pixels.size

    suma_total = np.dot(
        np.arange(256),
        histograma
    )

    peso_fondo = 0
    suma_fondo = 0
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
            peso_fondo *
            peso_objeto *
            (media_fondo - media_objeto) ** 2
        )

        if varianza > mejor_varianza:

            mejor_varianza = varianza
            umbral = i

    return int(umbral)


# =====================================================
# 4. CENTRAR IMAGEN POR CENTRO DE MASA
# =====================================================

def centrar_digito(imagen):

    array = np.asarray(
        imagen,
        dtype=np.float32
    )

    total = np.sum(array)

    if total <= 0:
        raise ValueError("El dígito está vacío.")

    yy, xx = np.indices(array.shape)

    centro_x = np.sum(xx * array) / total
    centro_y = np.sum(yy * array) / total

    desplazamiento_x = 13.5 - centro_x
    desplazamiento_y = 13.5 - centro_y

    imagen_centrada = imagen.transform(
        (28, 28),
        Image.Transform.AFFINE,
        (
            1, 0, -desplazamiento_x,
            0, 1, -desplazamiento_y
        ),
        resample=Image.Resampling.BILINEAR,
        fillcolor=0
    )

    return imagen_centrada


# =====================================================
# 5. PREPROCESAMIENTO
# =====================================================

def procesar_imagen(imagen_original):

    # Convertir a escala de grises
    gris = ImageOps.grayscale(imagen_original)

    # Mejorar contraste
    gris = ImageOps.autocontrast(gris)

    pixels = np.asarray(gris)

    # Detectar el color del fondo utilizando los bordes
    borde = np.concatenate([
        pixels[0, :],
        pixels[-1, :],
        pixels[:, 0],
        pixels[:, -1]
    ])

    # MNIST utiliza fondo negro y dígito blanco
    if np.median(borde) < 127:
        gris = ImageOps.invert(gris)

    # Calcular umbral Otsu
    umbral = calcular_umbral_otsu(gris)

    # Crear imagen binaria
    mascara = gris.point(
        lambda p: 255 if p > umbral else 0
    )

    # Encontrar límites del dígito
    bbox = mascara.getbbox()

    if bbox is None:
        raise ValueError(
            "No se detectó ningún dígito. "
            "Intenta subir otra imagen."
        )

    # Recortar utilizando la máscara
    digito = mascara.crop(bbox)

    ancho, alto = digito.size

    # Evitar deformaciones
    escala = min(
        20 / ancho,
        20 / alto
    )

    nuevo_ancho = max(
        1, round(ancho * escala)
    )

    nuevo_alto = max(
        1, round(alto * escala)
    )

    digito = digito.resize(
        (nuevo_ancho, nuevo_alto),
        Image.Resampling.LANCZOS
    )

    # Crear lienzo de 28x28
    lienzo = Image.new(
        "L",
        (28, 28),
        0
    )

    x = (28 - nuevo_ancho) // 2
    y = (28 - nuevo_alto) // 2

    lienzo.paste(
        digito,
        (x, y)
    )

    # Centrar mediante centro de masa
    lienzo = centrar_digito(lienzo)

    # Normalizar valores entre 0 y 1
    array = np.asarray(
        lienzo,
        dtype=np.float32
    ) / 255.0

    # Preparar entrada según la arquitectura del modelo
    forma_entrada = model.input_shape

    if isinstance(forma_entrada, list):
        raise ValueError(
            "Este código requiere un modelo con una sola entrada."
        )

    if len(forma_entrada) == 4:

        entrada = array[np.newaxis, ..., np.newaxis]

    elif len(forma_entrada) == 3:

        entrada = array[np.newaxis, ...]

    else:

        raise ValueError(
            f"Arquitectura de entrada no compatible: {forma_entrada}"
        )

    return lienzo, entrada, mascara


# =====================================================
# 6. SUBIR IMAGEN
# =====================================================

st.divider()

st.markdown("### 📤 Subir imagen")

archivo = st.file_uploader(
    "Selecciona una imagen con un solo dígito",
    type=["png", "jpg", "jpeg"],
    help="Se recomienda una imagen con un número claramente visible."
)


# =====================================================
# 7. PROCESAR Y RECONOCER
# =====================================================

if archivo is not None:

    try:

        # Abrir imagen
        archivo.seek(0)

        original = Image.open(archivo).convert("RGB")

        # Validar tamaño
        if original.width * original.height > 12_000_000:

            st.error("La imagen es demasiado grande.")
            st.stop()

        # Procesar imagen
        procesada, entrada, mascara = procesar_imagen(
            original
        )

        # ---------------------------------------------
        # MOSTRAR IMÁGENES
        # ---------------------------------------------

        st.divider()

        st.markdown("### 🖼️ Procesamiento de imagen")

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Imagen original**")

            st.image(
                original,
                caption="Imagen subida",
                width="stretch"
            )

        with col2:

            st.write("**Imagen procesada (28 × 28)**")

            imagen_ampliada = procesada.resize(
                (280, 280),
                Image.Resampling.NEAREST
            )

            st.image(
                imagen_ampliada,
                caption="Entrada de la red neuronal",
                width="stretch"
            )

        # ---------------------------------------------
        # MÁSCARA BINARIA
        # ---------------------------------------------

        with st.expander("Ver máscara de segmentación"):

            st.image(
                mascara,
                caption="Detección del dígito",
                width="stretch"
            )

        # ---------------------------------------------
        # DIAGNÓSTICO
        # ---------------------------------------------

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

            st.write(
                "Arquitectura esperada:",
                model.input_shape
            )

            st.write(
                "Tipo de datos:",
                str(entrada.dtype)
            )

        # ---------------------------------------------
        # PREDICCIÓN
        # ---------------------------------------------

        st.divider()

        st.markdown("### 🧠 Resultado del reconocimiento")

        resultado = model.predict(
            entrada,
            verbose=0
        )

        resultado = np.asarray(
            resultado[0],
            dtype=np.float64
        ).ravel()

        if len(resultado) != 10:

            raise ValueError(
                "El modelo debe tener 10 salidas, "
                "una por cada dígito del 0 al 9."
            )

        if not np.all(np.isfinite(resultado)):

            raise ValueError(
                "El modelo produjo valores no válidos."
            )

        # Comprobar si la salida ya contiene probabilidades
        suma = np.sum(resultado)

        if (
            np.all(resultado >= 0) and
            np.all(resultado <= 1) and
            np.isclose(suma, 1.0, atol=0.01)
        ):

            probabilidades = resultado

        else:

            # Convertir logits a probabilidades
            probabilidades = tf.nn.softmax(
                resultado
            ).numpy()

        # Dígito más probable
        numero = int(
            np.argmax(probabilidades)
        )

        confianza = float(
            probabilidades[numero] * 100
        )

        # ---------------------------------------------
        # RESULTADO PRINCIPAL
        # ---------------------------------------------

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

        # ---------------------------------------------
        # PROBABILIDADES
        # ---------------------------------------------

        st.markdown("### 📊 Probabilidad por dígito")

        probabilidades_dict = {
            str(i): float(probabilidades[i] * 100)
            for i in range(10)
        }

        st.bar_chart(
            probabilidades_dict,
            y_label="Probabilidad (%)"
        )

        # ---------------------------------------------
        # TRES MEJORES PREDICCIONES
        # ---------------------------------------------

        st.markdown("### 🔍 Tres predicciones principales")

        mejores = np.argsort(
            probabilidades
        )[-3:][::-1]

        for i in mejores:

            porcentaje = float(
                probabilidades[i] * 100
            )

            st.write(
                f"**Dígito {i}:** {porcentaje:.2f}%"
            )

            st.progress(
                float(probabilidades[i])
            )

    except Exception as e:

        st.error(
            f"Error al procesar la imagen: {e}"
        )


# =====================================================
# 8. PIE DE PÁGINA
# =====================================================

st.divider()

st.caption(
    "TecnoForms | Sistema de clasificación "
    "mediante Redes Neuronales Convolucionales (CNN)"
)
