
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image, ImageOps
from pathlib import Path
from io import BytesIO

# ==========================================
# CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="TecnoForms de Jhordan (Trabajo Final)",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 TecnoForms de Jhordan")
st.subheader("Trabajo Final | Clasificación de Dígitos Manuscritos")

st.write(
    "Sistema de reconocimiento automático de números manuscritos "
    "mediante redes neuronales convolucionales (CNN)."
)

st.divider()

# ==========================================
# CARGAR MODELO
# ==========================================

@st.cache_resource
def cargar_modelo():
    ruta = Path(__file__).parent / "modelo_digitos.keras"

    if not ruta.exists():
        raise FileNotFoundError("No se encontró modelo_digitos.keras")

    modelo = tf.keras.models.load_model(ruta, compile=False)

    if tuple(modelo.input_shape[1:]) != (28, 28, 1):
        raise ValueError("El modelo no tiene la entrada esperada de MNIST.")

    return modelo


try:
    model = cargar_modelo()
    st.success("Sistema listo. Modelo cargado correctamente.")

except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()


# ==========================================
# PREPROCESAMIENTO
# ==========================================

def procesar_imagen(imagen):

    # Convertir a escala de grises
    imagen = ImageOps.grayscale(imagen)
    imagen = ImageOps.autocontrast(imagen)

    pixeles = np.array(imagen)

    # Detectar el color del fondo mediante los bordes
    bordes = np.concatenate([
        pixeles[0, :],
        pixeles[-1, :],
        pixeles[:, 0],
        pixeles[:, -1]
    ])

    # MNIST: fondo negro y número blanco
    if np.median(bordes) > 127:
        imagen = ImageOps.invert(imagen)

    pixeles = np.array(imagen)

    # Detectar los trazos
    umbral = max(30, otsu(imagen))

    mascara = imagen.point(
        lambda p: 255 if p > umbral else 0
    )

    caja = mascara.getbbox()

    if caja is None:
        raise ValueError("No se detectó ningún número.")

    # Recortar el dígito
    digito = imagen.crop(caja)

    ancho, alto = digito.size

    # Mantener proporciones y ajustar a 20x20
    escala = min(20 / ancho, 20 / alto)

    nuevo_ancho = max(1, round(ancho * escala))
    nuevo_alto = max(1, round(alto * escala))

    digito = digito.resize(
        (nuevo_ancho, nuevo_alto),
        Image.Resampling.LANCZOS
    )

    # Crear imagen MNIST de 28x28
    lienzo = Image.new("L", (28, 28), 0)

    x = (28 - nuevo_ancho) // 2
    y = (28 - nuevo_alto) // 2

    lienzo.paste(digito, (x, y))

    # Centrar según el centro de masa
    array = np.array(lienzo, dtype=np.float32)

    masa = array.sum()

    if masa == 0:
        raise ValueError("La imagen está vacía.")

    yy, xx = np.indices(array.shape)

    cx = (array * xx).sum() / masa
    cy = (array * yy).sum() / masa

    desplazamiento_x = round(13.5 - cx)
    desplazamiento_y = round(13.5 - cy)

    centrada = Image.new("L", (28, 28), 0)

    centrada.paste(
        lienzo,
        (desplazamiento_x, desplazamiento_y)
    )

    # Normalizar entre 0 y 1
    entrada = np.array(
        centrada,
        dtype=np.float32
    ) / 255.0

    entrada = entrada.reshape(1, 28, 28, 1)

    return centrada, entrada


# ==========================================
# UMBRAL DE OTSU
# ==========================================

def otsu(imagen):

    pixeles = np.array(imagen)

    histograma = np.bincount(
        pixeles.ravel(),
        minlength=256
    )

    total = pixeles.size
    suma_total = np.dot(np.arange(256), histograma)

    peso_fondo = 0
    suma_fondo = 0
    mejor_varianza = -1
    mejor_umbral = 127

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
            mejor_umbral = i

    return mejor_umbral


# ==========================================
# SUBIR IMAGEN
# ==========================================

st.subheader("1. Subir imagen")

st.write("Selecciona una imagen que contenga un solo dígito.")

archivo = st.file_uploader(
    "Seleccionar imagen",
    type=["png", "jpg", "jpeg"]
)


# ==========================================
# RECONOCIMIENTO
# ==========================================

if archivo is not None:

    try:

        if len(archivo.getvalue()) > 10 * 1024 * 1024:
            raise ValueError("La imagen supera los 10 MB.")

        original = Image.open(
            BytesIO(archivo.getvalue())
        )

        original.load()

        if original.width * original.height > 12_000_000:
            raise ValueError("La resolución de la imagen es demasiado grande.")

        original = ImageOps.exif_transpose(original)

        # Mostrar original antes de procesar
        st.subheader("2. Imagen original")

        st.image(
            original,
            caption="Imagen subida por el usuario",
            use_container_width=True
        )

        # Procesamiento
        procesada, entrada = procesar_imagen(original)

        st.subheader("3. Imagen procesada")

        st.image(
            procesada.resize(
                (280, 280),
                Image.Resampling.NEAREST
            ),
            caption="Imagen adaptada a 28 × 28 píxeles",
            use_container_width=False
        )

        # Predicción
        st.subheader("4. Resultado")

        predicciones = model.predict(
            entrada,
            verbose=0
        )[0]

        if predicciones.shape != (10,) or not np.isfinite(predicciones).all():
            raise ValueError("El modelo produjo resultados no válidos.")

        numero = int(np.argmax(predicciones))
        confianza = float(predicciones[numero] * 100)

        st.success(f"### Número reconocido: {numero}")

        st.metric(
            "Confianza de la predicción",
            f"{confianza:.2f}%"
        )

        if confianza >= 80:
            st.success("Confianza alta.")
        elif confianza >= 50:
            st.warning("Confianza moderada. Verifica el resultado.")
        else:
            st.error("Confianza baja. Se recomienda revisar la imagen.")

        # Probabilidades
        st.subheader("5. Probabilidades por número")

        datos = {
            str(i): float(predicciones[i] * 100)
            for i in range(10)
        }

        st.bar_chart(datos)

        # Descargar imagen procesada
        buffer = BytesIO()
        procesada.save(buffer, format="PNG")

        st.download_button(
            "Descargar imagen procesada",
            data=buffer.getvalue(),
            file_name="imagen_procesada.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")


# ==========================================
# PIE DE PÁGINA
# ==========================================

st.divider()

st.caption(
    "TecnoForms de Jhordan | Trabajo Final | "
    "Python · TensorFlow · Keras · Streamlit"
)
