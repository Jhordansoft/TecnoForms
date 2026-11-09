
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image, ImageOps
from pathlib import Path
from io import BytesIO


# =====================================================
# 1. CONFIGURACIÓN GENERAL
# =====================================================

st.set_page_config(
    page_title="TecnoForms de Jhordan | Trabajo Final",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "modelo_digitos.keras"


# =====================================================
# 2. ESTILOS CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #172554 0%, transparent 35%),
        #080d19;
    color: #f1f5f9;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #172554, #172044, #102b3b);
    border: 1px solid #334b78;
    padding: 32px;
    border-radius: 20px;
    margin-bottom: 25px;
}

.brand {
    color: #67e8f9;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero h1 {
    color: white;
    font-size: clamp(30px, 5vw, 46px);
    font-weight: 800;
    margin: 12px 0;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
    margin-bottom: 0;
}

.section-title {
    font-size: 22px;
    font-weight: 800;
    color: #f8fafc;
    margin: 25px 0 14px;
}

.panel {
    background: #111b2d;
    border: 1px solid #27364d;
    border-radius: 16px;
    padding: 20px;
}

.result-card {
    background: linear-gradient(135deg, #102e3d, #13283b);
    border: 1px solid #256477;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
}

.result-number {
    font-size: 90px;
    font-weight: 800;
    color: #67e8f9;
    line-height: 1.2;
}

.result-label {
    color: #cbd5e1;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.confidence {
    color: #a5f3fc;
    font-size: 17px;
    font-weight: 700;
    margin-top: 8px;
}

div[data-testid="stFileUploader"] {
    background: #111b2d;
    border: 1px dashed #4b6588;
    border-radius: 15px;
    padding: 15px;
}

div[data-testid="stFileUploader"] label {
    color: white;
}

.stButton button {
    background: linear-gradient(90deg, #0891b2, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    width: 100%;
    min-height: 45px;
}

.stButton button:hover {
    border: 1px solid #67e8f9;
    color: white;
}

div[data-testid="stMetric"] {
    background: #111b2d;
    border: 1px solid #27364d;
    border-radius: 13px;
    padding: 15px;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. ENCABEZADO
# =====================================================

st.markdown("""
<div class="hero">

    <div class="brand">Inteligencia Artificial · Deep Learning</div>

    <h1>TECNOFORMS</h1>

    <p>
        <strong>de Jhordan (Trabajo Final)</strong>
        <br><br>
        Sistema inteligente de reconocimiento y clasificación
        de dígitos manuscritos mediante redes neuronales convolucionales.
    </p>

</div>
""", unsafe_allow_html=True)


# =====================================================
# 4. CARGA DEL MODELO
# =====================================================

@st.cache_resource(show_spinner="Cargando red neuronal...")
def load_trained_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró modelo_digitos.keras. "
            "Comprueba que esté en el repositorio."
        )

    loaded_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    expected_shape = loaded_model.input_shape

    if len(expected_shape) != 4 or tuple(expected_shape[1:]) != (28, 28, 1):
        raise ValueError(
            f"Arquitectura incompatible. Entrada esperada: "
            f"(None, 28, 28, 1). Entrada encontrada: {expected_shape}"
        )

    return loaded_model


try:
    model = load_trained_model()

    st.success("● Sistema operativo: red neuronal cargada correctamente")

except Exception as error:
    st.error("No fue posible iniciar el sistema.")
    st.code(str(error))
    st.info(
        "Verifica que modelo_digitos.keras exista y que la versión "
        "de TensorFlow sea compatible con la utilizada durante el entrenamiento."
    )
    st.stop()


# =====================================================
# 5. UMBRAL AUTOMÁTICO DE OTSU
# =====================================================

def otsu_threshold(image):

    pixels = np.asarray(image, dtype=np.uint8)

    histogram = np.bincount(
        pixels.ravel(),
        minlength=256
    )

    total = pixels.size
    total_sum = np.dot(np.arange(256), histogram)

    background_weight = 0
    background_sum = 0

    best_variance = -1
    best_threshold = 127

    for threshold in range(256):

        background_weight += histogram[threshold]

        if background_weight == 0:
            continue

        foreground_weight = total - background_weight

        if foreground_weight == 0:
            break

        background_sum += threshold * histogram[threshold]

        background_mean = background_sum / background_weight

        foreground_mean = (
            total_sum - background_sum
        ) / foreground_weight

        variance = (
            background_weight *
            foreground_weight *
            (background_mean - foreground_mean) ** 2
        )

        if variance > best_variance:

            best_variance = variance
            best_threshold = threshold

    return best_threshold


# =====================================================
# 6. CENTRADO DEL DÍGITO
# =====================================================

def center_digit(canvas):

    array = np.asarray(canvas, dtype=np.float32)

    total_mass = array.sum()

    if total_mass <= 0:
        raise ValueError("No se detectaron trazos en la imagen.")

    y_grid, x_grid = np.indices(array.shape)

    center_x = (array * x_grid).sum() / total_mass
    center_y = (array * y_grid).sum() / total_mass

    shift_x = int(round(13.5 - center_x))
    shift_y = int(round(13.5 - center_y))

    shifted = Image.new("L", (28, 28), 0)

    shifted.paste(
        canvas,
        (shift_x, shift_y)
    )

    return shifted


# =====================================================
# 7. PREPROCESAMIENTO DE IMAGEN
# =====================================================

def preprocess_image(original):

    # Manejar transparencias y convertir a escala de grises
    if original.mode in ("RGBA", "LA"):

        rgba = original.convert("RGBA")

        background = Image.new(
            "RGBA",
            rgba.size,
            (255, 255, 255, 255)
        )

        original = Image.alpha_composite(
            background,
            rgba
        ).convert("RGB")

    else:
        original = original.convert("RGB")

    gray = ImageOps.grayscale(original)

    # Mejorar contraste
    gray = ImageOps.autocontrast(gray)

    pixels = np.asarray(gray)

    # Analizar los bordes para determinar el fondo
    border = np.concatenate([
        pixels[0, :],
        pixels[-1, :],
        pixels[:, 0],
        pixels[:, -1]
    ])

    # Adaptar la polaridad al formato MNIST:
    # fondo negro y trazos blancos
    if np.median(border) > 127:
        gray = ImageOps.invert(gray)

    # Umbral automático
    threshold = otsu_threshold(gray)

    # Máscara para localizar los trazos
    mask = gray.point(
        lambda value: 255 if value > threshold else 0
    )

    bbox = mask.getbbox()

    if bbox is None:
        raise ValueError(
            "No se pudo detectar ningún número. "
            "Sube una imagen con mejor iluminación y contraste."
        )

    # Recortar el dígito
    digit = gray.crop(bbox)

    width, height = digit.size

    if width < 2 or height < 2:
        raise ValueError("El número detectado es demasiado pequeño.")

    # Mantener las proporciones y ajustar a 20x20
    scale = min(
        20 / width,
        20 / height
    )

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    digit = digit.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Lienzo de 28x28 con fondo negro
    canvas = Image.new(
        "L",
        (28, 28),
        0
    )

    x = (28 - new_width) // 2
    y = (28 - new_height) // 2

    canvas.paste(digit, (x, y))

    # Centrar según la distribución de los píxeles
    canvas = center_digit(canvas)

    # Normalización entre 0 y 1
    array = np.asarray(
        canvas,
        dtype=np.float32
    ) / 255.0

    # Dimensiones compatibles con CNN
    array = np.expand_dims(
        array,
        axis=(0, -1)
    )

    return canvas, array


# =====================================================
# 8. INFORMACIÓN DEL SISTEMA
# =====================================================

st.markdown("""
<div class="section-title">Panel de reconocimiento</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Modelo", "CNN")

with col2:
    st.metric("Clases", "10 dígitos")

with col3:
    st.metric("Entrada", "28 × 28 px")


# =====================================================
# 9. CARGA DE IMAGEN
# =====================================================

st.markdown("""
<div class="section-title">01 · Seleccionar imagen</div>
""", unsafe_allow_html=True)

st.write(
    "Carga una imagen que contenga un solo número manuscrito. "
    "Se recomienda utilizar tinta oscura sobre papel claro, "
    "con buena iluminación y sin sombras."
)

uploaded_file = st.file_uploader(
    "Selecciona tu imagen",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False
)


# =====================================================
# 10. PROCESAMIENTO Y PREDICCIÓN
# =====================================================

if uploaded_file is not None:

    try:

        if len(uploaded_file.getvalue()) > 10 * 1024 * 1024:
            raise ValueError(
                "El archivo supera el límite de 10 MB."
            )

        # Abrir imagen de forma segura
        original = Image.open(
            BytesIO(uploaded_file.getvalue())
        )

        original.load()

        if original.width * original.height > 12_000_000:
            raise ValueError(
                "La imagen tiene demasiados píxeles. "
                "Utiliza una imagen de menor resolución."
            )

        # Procesar imagen
        processed_image, input_array = preprocess_image(original)

        st.markdown("""
        <div class="section-title">02 · Procesamiento de imagen</div>
        """, unsafe_allow_html=True)

        col_original, col_processed = st.columns(2)

        with col_original:

            st.markdown("**Imagen original**")

            st.image(
                original,
                caption=f"{original.width} × {original.height} píxeles",
                use_container_width=True
            )

        with col_processed:

            st.markdown("**Imagen procesada**")

            st.image(
                processed_image.resize(
                    (280, 280),
                    Image.Resampling.NEAREST
                ),
                caption="28 × 28 píxeles · Escala de grises",
                use_container_width=True
            )

        # Comprobar entrada
        if not np.isfinite(input_array).all():
            raise ValueError("La imagen contiene valores no válidos.")

        if input_array.shape != (1, 28, 28, 1):
            raise ValueError("Las dimensiones de entrada son incorrectas.")

        # Predicción
        with st.spinner("Analizando características del dígito..."):

            predictions = model.predict(
                input_array,
                verbose=0
            )[0]

        if predictions.shape != (10,):
            raise ValueError("El modelo no devuelve diez clases.")

        if not np.isfinite(predictions).all():
            raise ValueError("El modelo generó resultados no válidos.")

        if np.any(predictions < 0) or np.any(predictions > 1):
            raise ValueError("Las probabilidades generadas no son válidas.")

        if not np.isclose(np.sum(predictions), 1.0, atol=0.02):
            raise ValueError("La salida del modelo no tiene probabilidades normalizadas.")

        # Obtener resultado
        predicted_digit = int(np.argmax(predictions))

        confidence = float(
            predictions[predicted_digit] * 100
        )

        # =================================================
        # 11. RESULTADO
        # =================================================

        st.markdown("""
        <div class="section-title">03 · Resultado del reconocimiento</div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">

            <div class="result-label">Dígito identificado</div>

            <div class="result-number">{predicted_digit}</div>

            <div class="confidence">
                Confianza del modelo: {confidence:.2f}%
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        if confidence >= 80:

            st.success(
                "El modelo presenta una confianza alta en esta predicción."
            )

        elif confidence >= 50:

            st.warning(
                "La confianza es moderada. Se recomienda verificar el resultado."
            )

        else:

            st.error(
                "Confianza baja. El modelo no puede identificar "
                "este número con suficiente seguridad."
            )

        # =================================================
        # 12. PROBABILIDADES
        # =================================================

        st.markdown("""
        <div class="section-title">04 · Distribución de probabilidades</div>
        """, unsafe_allow_html=True)

        chart_data = {
            "Dígito": [str(i) for i in range(10)],
            "Probabilidad (%)": [
                float(predictions[i] * 100)
                for i in range(10)
            ]
        }

        st.bar_chart(
            chart_data,
            x="Dígito",
            y="Probabilidad (%)",
            height=300
        )

        # =================================================
        # 13. TRES MEJORES PREDICCIONES
        # =================================================

        st.markdown("""
        <div class="section-title">05 · Predicciones alternativas</div>
        """, unsafe_allow_html=True)

        best_predictions = np.argsort(
            predictions
        )[-3:][::-1]

        columns = st.columns(3)

        for index, digit in enumerate(best_predictions):

            with columns[index]:

                st.metric(
                    label=f"Posición {index + 1}",
                    value=str(int(digit)),
                    delta=f"{predictions[digit] * 100:.2f}%"
                )

        # =================================================
        # 14. DESCARGAR IMAGEN PROCESADA
        # =================================================

        buffer = BytesIO()

        processed_image.save(
            buffer,
            format="PNG"
        )

        st.download_button(
            label="Descargar imagen procesada",
            data=buffer.getvalue(),
            file_name="digito_procesado.png",
            mime="image/png",
            use_container_width=True
        )

    except Exception as error:

        st.error(
            f"No fue posible procesar la imagen: {error}"
        )

        st.info(
            "Intenta utilizar una imagen donde el número esté "
            "completo, centrado, bien iluminado y separado de otros elementos."
        )


# =====================================================
# 15. PIE DE PÁGINA
# =====================================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    padding:20px;
    color:#94a3b8;
">

    <div style="
        color:#67e8f9;
        font-size:18px;
        font-weight:800;
        letter-spacing:2px;
    ">
        TECNOFORMS
    </div>

    <p>
        de Jhordan · Trabajo Final
    </p>

    <p style="font-size:12px;">
        Clasificación de Dígitos Manuscritos
        <br>
        Desarrollado con Python, TensorFlow, Keras y Streamlit
    </p>

</div>
""", unsafe_allow_html=True)
