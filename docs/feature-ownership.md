# Feature ownership

## Propósito

El *feature ownership* define quién mantiene cada parte de SafeAgent y quién
realiza una segunda revisión antes de integrar cambios. El owner principal
coordina el mantenimiento de su feature, comprueba que los cambios incluyen
tests adecuados, mantiene su documentación y atiende las revisiones de los pull
requests. El reviewer aporta una revisión independiente y comprueba especialmente
los contratos y riesgos propios de esa feature.

Las asignaciones reflejan el estado visible del repositorio. El proyecto sigue
la organización de Cookiecutter Data Science, con datos versionados mediante
DVC, código bajo `taed2_safeagent/`, tests en `tests/` y documentación en
`docs/`. Los estados indican implementación observada, no una fecha de entrega.

## Resumen de asignaciones

| Feature / módulo | Componentes / archivos clave | Owner principal | Reviewer | Estado / milestone |
| --- | --- | --- | --- | --- |
| Ingesta del dataset fijado y auditoría de origen | `data/raw/shell_safety.dvc`, `taed2_safeagent/data/source.py`, `taed2_safeagent/data/audit.py`, `params.yaml` (`source`), `tests/test_source.py`, `tests/test_audit.py` | David González | Joel Márquez | Implementado: snapshot fijado, verificación de hash y auditoría |
| Agrupación, split y prevención de data leakage | `taed2_safeagent/data/grouping.py`, `taed2_safeagent/data/split.py`, `params.yaml` (`grouping`, `split`), `tests/test_grouping.py` | Joel Márquez | Pau González | Implementado: split por grupos train/validation/test |
| Contratos y validación de datos | `taed2_safeagent/data/inputs.py`, `taed2_safeagent/data/validation.py`, `taed2_safeagent/dataset.py`, `reports/data/`, `tests/test_validation.py` | Pau González | Pau Balaguer | Implementado: controles de filas, etiquetas, duplicados y cuarentena |
| Extracción de features y contexto de ejecución | `taed2_safeagent/features.py`, `taed2_safeagent/data/inputs.py`, `params.yaml` (`context`, `baseline.tfidf`), `tests/test_baseline.py` | Pau Balaguer | David González | Implementado: TF-IDF con n-gramas de caracteres y entrada `command-context`; n-gramas de palabra no observados |
| Baseline y pipeline DVC | `taed2_safeagent/modeling/train.py`, `taed2_safeagent/modeling/evaluate.py`, `dvc.yaml`, `dvc.lock`, `params.yaml` (`baseline`), `tests/test_baseline.py`, `tests/test_model_pipeline.py` | David González | Pau González | Implementado: SVM lineal; Logistic Regression no forma parte del baseline actual |
| Experimentos, encoders y tracking | `taed2_safeagent/modeling/encoder.py`, `experiment.py`, `tracking.py`, `register.py`, `params.yaml` (`tracking`, `encoders`), `docs/encoder-training.md`, `tests/test_encoder.py`, `tests/test_tracking.py`, `tests/test_registration.py` | Pau Balaguer | Joel Márquez | Implementado: MLflow en DagsHub, entrenamiento/registro de encoders |
| Quality assurance y sostenibilidad | `tests/`, `notebooks/kaggle_baseline.ipynb`, `.github/workflows/data.yml`, `pyproject.toml`, `Makefile`, `taed2_safeagent/modeling/energy.py`, informes de energía de `reports/` | Joel Márquez | David González | Implementado parcialmente: pytest, Ruff, Pylint, CI y CodeCarbon; PyNBLint está en dependencias, pero no se ejecuta en el workflow |
| Serving y despliegue | Diseño pendiente para `taed2_safeagent/` y documentación de despliegue | Pau González | Pau Balaguer | Pendiente: no se encuentran API FastAPI, schemas Pydantic, endpoints ALLOW/ASK/DENY ni configuración de Virtech en el código actual |

No se considera que una dependencia aparezca en `uv.lock` como prueba de que
una funcionalidad esté integrada: los estados se basan en código, configuración
del proyecto, pipeline y documentación operativa.
En particular, `params.yaml` fija actualmente la fuente
`tomngdev/shell-safety-v1.1`; no está configurada como `shell-safety-v2`.

## Responsabilidades por integrante

### David González

- **Ingesta y auditoría de datos:** mantener la descarga reproducible del
  snapshot fijado, sus hashes y conteos; conservar procedencia, deduplicación y
  cuarentena. En el informe, presentar la identidad de la fuente, el resultado
  de la auditoría y los límites conocidos del dataset.
- **Baseline y pipeline DVC:** mantener el entrenamiento/evaluación de
  referencia y las dependencias de sus stages. En el informe, registrar
  parámetros, métricas de validation, errores relevantes —en particular
  `DENY` predicho como `ALLOW`— y la versión de datos/modelo.

### Joel Márquez

- **Agrupación, split y prevención de leakage:** mantener las reglas de
  fingerprints, similitud y asignación por grupos; verificar que los grupos no
  cruzan particiones. En el informe, justificar el protocolo, semilla,
  proporciones y diagnósticos del split.
- **Quality assurance y sostenibilidad:** cuidar tests, lint, CI y las
  mediciones CodeCarbon disponibles. En el informe, documentar qué checks se
  ejecutaron, resultados, cobertura de CI y las limitaciones/incertidumbre de
  las estimaciones energéticas.

### Pau Balaguer

- **Extracción de features y contexto:** mantener la vectorización por
  caracteres y la construcción de entradas `command` y `command-context`.
  En el informe, describir el preprocesamiento y comparar los modos de entrada;
  cualquier experimento con n-gramas de palabra debe identificarse como una
  ampliación, no como comportamiento actual.
- **Experimentos, encoders y tracking:** mantener la reproducción, metadatos y
  registro de experimentos. En el informe, comparar los modelos y modos de
  entrada con el mismo split, y conservar la procedencia de código, datos y
  resultados.

### Pau González

- **Contratos y validación de datos:** mantener la validación de los ejemplos
  preparados y sus reportes. En el informe, explicar los invariantes, conteos
  esperados, filas duplicadas/cuarentenadas y limitaciones del contrato.
- **Serving y despliegue:** liderar el diseño futuro de una API y su despliegue.
  Antes de marcar esta feature como implementada, definir schemas, validación de
  request/response, comportamiento seguro para `ALLOW`/`ASK`/`DENY`, pruebas de
  inferencia y una estrategia verificable de despliegue en Virtech. No se debe
  asumir que estos componentes ya existen.

## Protocolo de Pull Requests y ownership

### Siete reglas de commit

El repositorio no contiene una lista anterior de siete reglas de commit. Para
evitar atribuirle normas no documentadas, esta sección fija un protocolo
alineado con el historial, la configuración y las prácticas de reproducción
actuales:

1. Trabajar en ramas con nombre descriptivo: `feature/<nombre>` para cambios
   funcionales y `docs/<nombre>` para cambios documentales.
2. Mantener cada commit centrado en un único cambio lógico; no mezclar una
   feature no relacionada con limpieza o resultados ajenos.
3. Usar mensajes de commit tipo Conventional Commits, por ejemplo
   `feat(model): ...`, `fix(data): ...`, `test: ...`, `docs: ...` o `chore: ...`.
4. No añadir credenciales, tokens ni secretos al repositorio ni a los mensajes
   de commit; usar variables de entorno o configuración local ignorada por Git.
5. No modificar manualmente la fuente raw fijada. Un cambio de fuente requiere
   actualizar explícitamente revisión, hashes, conteos y protocolo de datos.
6. Versionar los artefactos grandes con DVC y revisar los cambios de
   `dvc.lock`/punteros; mantener en Git los reportes pequeños previstos por el
   proyecto.
7. Antes de solicitar integración, ejecutar los tests y checks pertinentes.
   Para cambios de código, usar pytest, Ruff y Pylint; para cambios
   documentales, comprobar `mkdocs build --strict`.

### Revisión e integración

- Todo pull request debe indicar la feature y enlazar los cambios de
  documentación/tests correspondientes.
- En pull requests de ramas `feature/<nombre>`, la aprobación del owner
  principal de la feature afectada es obligatoria antes de integrar. El reviewer
  secundario asignado en la tabla realiza la segunda revisión; si el owner es
  quien propone el cambio, debe aprobarlo otro integrante con responsabilidad
  sobre la feature, acordado en el propio pull request.
- No se debe aprobar un cambio que debilite los controles de leakage,
  validación o seguridad de inferencia sin documentar y probar explícitamente
  el comportamiento nuevo.
- Esta tabla define el acuerdo del equipo; la aprobación automática de GitHub
  solo será obligatoria a nivel de repositorio si se configura una regla de
  protección de ramas.
