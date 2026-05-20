# OpenProctor

### 1. Resumen Ejecutivo

**OpenProctor** es una herramienta de software *open source* diseñada para automatizar la auditoría visual de exámenes en entornos educativos y de oposiciones. El sistema analiza grabaciones de pantalla de larga duración (en formato MP4, habitualmente capturadas con herramientas como **Veyon**) para identificar de forma autónoma posibles conductas de copia o el uso de software no autorizado (como Discord, WhatsApp Web, ChatGPT o GitHub Copilot), reduciendo drásticamente el tiempo de revisión de los docentes mediante un enfoque híbrido de Inteligencia Artificial.

### 2. El Problema (Contexto Real)

En evaluaciones digitalizadas, la monitorización en tiempo real es compleja. Grabar las pantallas de los alumnos genera un volumen masivo de datos: por ejemplo, una sola sesión con 20 alumnos genera **30 horas de vídeo**. Revisar manualmente este material para buscar infracciones de pocos segundos es humanamente inviable para un profesor.

### 3. La Solución Técnica: El Patrón de Embudo (Funnel Pattern)

Para resolver el enorme cuello de botella computacional que supondría analizar millones de imágenes con Inteligencia Artificial avanzada, el proyecto implementa un pipeline de optimización en tres fases consecutivas, yendo de lo más "barato" y rápido a lo más complejo:

1. **Fase de Reducción Temporal y Espacial:** Mediante propiedades nativas de archivos MP4 (OpenCV), el sistema salta en el tiempo y extrae solo un fotograma cada *X* segundos (ej. 5s). Acto seguido, aplica un algoritmo de similitud matemática (MSE / Hashing Perceptual). Si la pantalla apenas ha cambiado respecto al fotograma anterior, se descarta. Esto elimina hasta el 85% del metraje inútil (momentos en los que el alumno solo lee o piensa).
2. **Fase de Cribado Ligero (OCR Triage):** Los fotogramas supervivientes se procesan con un motor de reconocimiento óptico de caracteres (`EasyOCR`). El sistema busca mediante expresiones regulares palabras clave de herramientas prohibidas (*"discord", "gpt", "copilot", "chat"*, etc.). Solo los fotogramas con coincidencias de texto pasan a la fase final.
3. **Fase de Razonamiento Multimodal (VLM Local):** Las capturas altamente sospechosas son enviadas a un Modelo de Lenguaje Visual local (como `Moondream2` o `LLaVA` ejecutados en **Ollama**). El modelo actúa como el "cerebro" final, analizando el contexto visual para discriminar falsos positivos (por ejemplo, si la palabra "Discord" aparecía simplemente redactada en un PDF de los apuntes del examen) y confirmando la infracción en formato JSON.

### 4. Arquitectura del Sistema

El prototipo está estructurado de manera modular bajo buenas prácticas de ingeniería de software:

* `/data`: Almacenamiento local segmentado de imágenes intermedias, sospechosas y bases de datos de resultados.
* `/src`: Módulos independientes de procesamiento de vídeo, filtrado OCR y analítica VLM.
* `main.py`: Orquestador y tubería de ejecución (*pipeline*).
* `app.py`: Panel de control interactivo desarrollado en **Streamlit** que permite al docente cargar los vídeos, ver el progreso en tiempo real y auditar las infracciones confirmadas con marcas de tiempo exactas y la justificación de la IA.

### 5. Stack Tecnológico (100% Open Source y Local)

* **Lenguaje:** Python 3.10+
* **Procesamiento de Imagen/Vídeo:** OpenCV (`opencv-python`), Pillow
* **Extracción de Texto:** EasyOCR
* **Modelos de IA Visual:** Moondream2 / LLaVA (Orquestados localmente mediante la API de **Ollama**)
* **Interfaz Gráfica:** Streamlit

### 6. Valor Educativo (Por qué impartirlo en clase)

Este proyecto sirve como un caso de estudio excepcional para los estudiantes porque:

* **Enseña optimización de recursos:** Aprenden que "lanzarle IA a todo" es costoso e ineficiente, y que los algoritmos tradicionales combinados con IA son la clave de la ingeniería de software moderna.
* **Privacidad y soberanía de datos:** Al ejecutarse 100% en local mediante Ollama, los datos de los exámenes de los alumnos jamás suben a nubes de terceros (OpenAI, Google), cumpliendo estrictas normativas de protección de datos (RGPD).
* **Desarrollo Full-Stack de IA:** Conecta preprocesamiento de datos, lógica de IA multimodal y diseño de interfaz de usuario en un producto de software funcional de principio a fin.
