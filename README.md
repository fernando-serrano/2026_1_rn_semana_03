# Adaline AND

Este proyecto contiene un ejemplo simple del algoritmo **Adaline** aplicado a la tabla lógica **AND**.

## Objetivo

Entrenar una neurona lineal con el conjunto de verdad de AND, registrar el proceso en archivos de log y generar imágenes con Matplotlib.

## Flujo general

1. Se define la tabla lógica AND:
   - `[[0, 0], [0, 1], [1, 0], [1, 1]]`
2. Se inicializan los pesos y el sesgo de forma aleatoria.
3. Se entrena el modelo durante varias épocas.
4. Se calcula el error cuadrático medio en cada época.
5. Se guardan los logs del entrenamiento en `log/`.
6. Se generan imágenes de salida en `log/images/`.

## Entradas

El script usa como entrada principal la tabla AND:

- `x1 = 0, x2 = 0 -> 0`
- `x1 = 0, x2 = 1 -> 0`
- `x1 = 1, x2 = 0 -> 0`
- `x1 = 1, x2 = 1 -> 1`

También usa parámetros internos simples:

- `learning_rate = 0.1`
- `epochs = 100`
- `seed = 42`

## Salidas

Al ejecutar el script se generan estas salidas:

- Un archivo de log en `log/adaline_and_YYYYMMDD_HHMMSS.log`
- Una imagen de la pérdida en `log/images/loss_YYYYMMDD_HHMMSS.png`
- Una imagen de la frontera de decisión en `log/images/boundary_YYYYMMDD_HHMMSS.png`
- Mensajes en consola con la ruta de los archivos generados

## Estructura del proyecto

```text
semana_03/
├── adaline_and.py
├── requirements.txt
├── README.md
└── log/
    └── images/
```

## Requisitos

Instalar dependencias con:

```bash
pip install -r requirements.txt
```

## Ejecución

Desde la carpeta `semana_03`:

```bash
python adaline_and.py
```

Si usas el entorno virtual del proyecto en Windows, puedes ejecutar con el intérprete del `.venv`.

## Notas

- El proyecto está pensado como una base simple para entender el flujo de Adaline con AND.
- Los archivos dentro de `log/` se regeneran en cada ejecución con marcas de tiempo.