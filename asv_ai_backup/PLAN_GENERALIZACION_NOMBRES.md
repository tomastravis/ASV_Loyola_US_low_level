# Plan de Generalización de Nombres - ASV AI Package

## Objetivo
Eliminar referencias específicas a "ASV" (Autonomous Surface Vehicle) para hacer el código más general y aplicable a cualquier tipo de robot o agente autónomo.

## Motivación
Aunque el sistema fue diseñado inicialmente para barcos autónomos (ASV), la arquitectura es suficientemente general para aplicarse a otros tipos de robots. El renombrado facilitará la reutilización del código en diferentes dominios.

---

## 🎯 FASE 1: Renombrado de Carpetas y Estructura de Paquete

### 1.1 Nombre del Paquete Principal
**Estado Actual:** `asv_ai`  
**Propuesta:** `rl_multi_agent` o `ppo_multi_robot`  
**Justificación:** Enfatiza el enfoque en aprendizaje por refuerzo multi-agente

**Archivos Afectados:**
- `setup.py` (línea 6: package_name)
- `package.xml` (nombre del paquete)
- `resource/asv_ai` → `resource/rl_multi_agent`
- Launch files (líneas que referencian 'asv_ai')
- Todos los imports en archivos Python

### 1.2 Submódulos/Carpetas
| Carpeta Actual | Nombre Propuesto | Justificación |
|----------------|------------------|---------------|
| `asv_agent/` | `robot_agent/` | Agente físico genérico |
| `asv_env/` | `environment/` o `env_manager/` | Gestor del entorno genérico |
| `asv_path/` | `path/` o `trajectory/` | Trayectorias genéricas |
| `asv_ppo/` | `ppo/` | Ya es general, solo quitar prefijo |

**Impacto:** 
- Todos los imports relativos dentro del paquete
- Entry points en `setup.py`
- Referencias en launch files

---

## 🎯 FASE 2: Renombrado de Archivos Python

### 2.1 Archivos de Nodos ROS2
| Archivo Actual | Nombre Propuesto | Ubicación Nueva |
|----------------|------------------|-----------------|
| `asv_agent_node.py` | `robot_agent_node.py` | `robot_agent/` |
| `asv_agent.py` | `robot_agent.py` o `physical_model.py` | `robot_agent/` |
| `asv_env_node.py` | `env_node.py` | `environment/` |
| `asv_path.py` | `parametrized_path.py` | `trajectory/` |
| `ppo_node.py` | ✅ Ya genérico | `ppo/` |

### 2.2 Entry Points en setup.py
**Actual (líneas 28-30):**
```python
'asv_env_node = asv_ai.asv_env.asv_env_node:main',
'asv_agent_node = asv_ai.asv_agent.asv_agent_node:main',
'ppo_node = asv_ai.asv_ppo.ppo_node:main',
```

**Propuesta:**
```python
'env_node = rl_multi_agent.environment.env_node:main',
'robot_agent_node = rl_multi_agent.robot_agent.robot_agent_node:main',
'ppo_node = rl_multi_agent.ppo.ppo_node:main',
```

---

## 🎯 FASE 3: Renombrado de Clases Python

### 3.1 Clases Principales
| Clase Actual | Nombre Propuesto | Archivo |
|--------------|------------------|---------|
| `ASVAgent` | `PhysicalRobotModel` o `RobotDynamics` | `robot_agent.py` |
| `ASVAgentNode` | `RobotAgentNode` | `robot_agent_node.py` |
| `ASVEnvNode` | `EnvironmentNode` | `env_node.py` |
| `ParametrizedPath` | ✅ Ya genérico | `parametrized_path.py` |
| `PPONode` | ✅ Ya genérico | `ppo_node.py` |

### 3.2 Justificación de Nombres
- **`PhysicalRobotModel`**: Enfatiza que modela la dinámica física de cualquier robot
- **`RobotAgentNode`**: Nodo ROS2 que maneja un agente robótico
- **`EnvironmentNode`**: Nodo que gestiona el entorno (recompensas, observaciones)

---

## 🎯 FASE 4: Renombrado de Topics y Nombres de Nodos ROS2

### 4.1 Nombres de Nodos
| Nodo Actual | Nombre Propuesto |
|-------------|------------------|
| `asv_agent_node` | `robot_agent_node` |
| `asv_env_node` | `environment_node` o `env_node` |
| `ppo_node` | ✅ Ya genérico |

### 4.2 Topics ROS2
**En `asv_env_node.py` (líneas 32-38):**
| Topic Actual | Nombre Propuesto |
|--------------|------------------|
| `/asv_env/path_marker` | `/environment/path_marker` |
| `/asv_env/centroid_marker` | `/environment/centroid_marker` |
| `/asv_env/core_boundary` | `/environment/core_boundary` |
| `/asv_env/safety_boundary` | `/environment/safety_boundary` |
| `/asv_env/warning_boundary` | `/environment/warning_boundary` |

### 4.3 TF Frames
**En `asv_agent_node.py` (línea 100):**
```python
# Actual
t.child_frame_id = f'ASV{self.agent_id}/base_link'

# Propuesta
t.child_frame_id = f'robot_{self.agent_id}/base_link'
```

---

## 🎯 FASE 5: Launch Files

### 5.1 Archivos Launch
| Archivo Actual | Nombre Propuesto |
|----------------|------------------|
| `asv_system.launch.py` | `multi_robot_system.launch.py` |
| `asv_train.launch.py` | `train_ppo.launch.py` |

### 5.2 Parámetros y Descripciones
**Cambios en descripciones:**
```python
# Actual
'Number of ASV agents'

# Propuesta
'Number of robot agents'
```

**Directorio de rollouts (línea 33 en asv_train.launch.py):**
```python
# Actual
'rollout_dir', default_value='~/Desktop/ASV_Rollouts'

# Propuesta
'rollout_dir', default_value='~/Desktop/RL_Rollouts'
```

### 5.3 Referencias a Ejecutables
```python
# Actual
executable='asv_agent_node',
name=f'asv_agent_node_{i}',

# Propuesta
executable='robot_agent_node',
name=f'robot_agent_node_{i}',
```

---

## 🎯 FASE 6: Strings y Mensajes en el Código

### 6.1 Mensajes de Log
| Ubicación | Actual | Propuesto |
|-----------|--------|-----------|
| `ppo_node.py:192` | `'ASV PPO Node started with {self.num_agents} agents'` | `'PPO Node started with {self.num_agents} agents'` |
| `asv_env_node.py:101` | `'ASV Environment Node started with {self.num_agents} agents'` | `'Environment Node started with {self.num_agents} agents'` |
| `asv_agent_node.py:48` | `'ASV Agent Node {self.agent_id} started...'` | `'Robot Agent Node {self.agent_id} started...'` |

### 6.2 Nombres de Archivos Generados
**En `ppo_node.py` (línea 205):**
```python
# Actual
f"asv_formation_{self.session_id}.json"

# Propuesta
f"formation_{self.session_id}.json"
```

**Rutas por defecto (líneas 31, 838, 839):**
```python
# Actual
'~/Desktop/ASV_Rollouts'

# Propuesta
'~/Desktop/RL_Rollouts'
```

### 6.3 Comentarios en el Código
**Comentarios específicos a ASV que generalizar:**
- `asv_agent_node.py:21`: "Initialize physical ASV model" → "Initialize physical robot model"
- `asv_agent_node.py:55`: "Convert ROS state to ASV model format" → "Convert ROS state to robot model format"
- `asv_agent.py:9`: "ASV According to state" → "Robot physical model"
- `data_conversion.py:72,74`: Comentarios sobre "ASV model" → "robot model"

---

## 🎯 FASE 7: Variables y Atributos Internos

### 7.1 Variables que Referencian ASV
| Ubicación | Variable Actual | Propuesta |
|-----------|----------------|-----------|
| `asv_agent_node.py:23` | `self.asv_model` | `self.robot_model` |
| `asv_agent_node.py:56-57` | `asv_state` | `robot_state` |
| `asv_agent_node.py` | `_convert_ros_to_asv_state()` | `_convert_ros_to_robot_state()` |
| `asv_agent_node.py` | `_convert_asv_to_ros_state()` | `_convert_robot_to_ros_state()` |
| `asv_agent_node.py` | `_action_to_asv_force()` | `_action_to_robot_force()` |

### 7.2 Parámetros y Constantes
**En `asv_agent.py` (línea 49):**
```python
# Actual
self.id = id # Identificador del ASV

# Propuesta
self.id = id # Robot identifier
```

**Comentarios en español → inglés (opcional pero recomendado):**
- "Identificador del ASV" → "Robot identifier"
- "Simular el movimiento del ASV" → "Simulate robot movement"
- "Distancia entre el ASV y un punto" → "Distance between robot and point"

---

## 🎯 FASE 8: Archivos de Configuración y Recursos

### 8.1 RViz Config
| Archivo Actual | Nombre Propuesto |
|----------------|------------------|
| `rviz/asv.rviz` | `rviz/multi_robot.rviz` |

### 8.2 Referencias en Launch Files
```python
# Actual (línea 89 en asv_system.launch.py)
FindPackageShare('asv_ai'), 'rviz', 'asv.rviz'

# Propuesta
FindPackageShare('rl_multi_agent'), 'rviz', 'multi_robot.rviz'
```

### 8.3 Test Files
**En `test_model_loading.py` (línea 10, 13):**
```python
# Actual
sys.path.append('/asv_ws/src/asv_ai')
test_dir = "/tmp/test_asv_models"

# Propuesta
sys.path.append('/rl_ws/src/rl_multi_agent')
test_dir = "/tmp/test_rl_models"
```

---

## 🎯 FASE 9: Descripciones y Documentación

### 9.1 setup.py
```python
# Actual (línea 23)
description='ASV AI package for controlling autonomous surface vehicles',

# Propuesta
description='Multi-robot reinforcement learning package using PPO',
```

### 9.2 README y Documentación
- Actualizar README.md si existe
- Documentación interna en archivos markdown
- Comentarios de módulos (docstrings)

---

## 📋 RESUMEN DE IMPACTO

### Archivos que DEBEN modificarse (orden recomendado):
1. ✅ **Crear backup** de toda la carpeta `asv_ai/`
2. 📦 **Renombrar carpeta principal**: `asv_ai/` → `rl_multi_agent/`
3. 📁 **Renombrar subcarpetas** (4 carpetas)
4. 📄 **Renombrar archivos Python** (4 archivos principales)
5. 🔧 **Actualizar `setup.py`** (package_name, entry_points, description)
6. 📦 **Actualizar `package.xml`** (nombre del paquete)
7. 🚀 **Actualizar launch files** (2 archivos: ejecutables, nombres, paths)
8. 🐍 **Actualizar clases Python** (3 clases principales)
9. 🔄 **Actualizar imports** en todos los archivos Python
10. 📝 **Actualizar topics ROS2** en env_node
11. 🌐 **Actualizar TF frames** en robot_agent_node
12. 💬 **Actualizar strings/logs** en todos los archivos
13. 🔧 **Actualizar variables internas** (self.asv_model, etc.)
14. 📊 **Actualizar archivo RViz**
15. 🧪 **Actualizar tests**

### Estimación de Cambios:
- **Archivos Python modificados**: ~8-10 archivos
- **Launch files modificados**: 2 archivos
- **Carpetas renombradas**: 5 (principal + 4 subcarpetas)
- **Archivos renombrados**: ~6 archivos
- **Clases renombradas**: 3 clases principales
- **Variables/métodos renombrados**: ~15-20 referencias

---

## ⚠️ CONSIDERACIONES IMPORTANTES

### 1. **Compatibilidad con URDF**
En `asv_train.launch.py` (línea 99):
```python
'asv_loyola.urdf.xacro'  # Archivo físico del robot real
```
**Decisión:** Este archivo URDF es específico del hardware real (barcos ASV Loyola) y **NO debe renombrarse**. Es correcto mantener el nombre original porque describe el robot físico real.

### 2. **Referencias a Paquetes Externos**
```python
FindPackageShare('yf_description')  # Paquete externo, no modificar
```
Paquetes como `yf_description` son dependencias externas y no deben modificarse.

### 3. **Parámetros Físicos**
En `data_conversion.py` (líneas 72, 74, 85, 87), los valores numéricos (0.5, 2.3, etc.) son específicos del modelo físico ASV. 
**Opciones:**
- **Opción A (Recomendada):** Parametrizar estos valores y leerlos de un archivo de configuración
- **Opción B:** Mantenerlos como están con comentarios claros que indican que son valores ejemplo

### 4. **Rollback Plan**
Si algo falla después de los cambios:
1. Restaurar backup de `asv_ai/`
2. Ejecutar `colcon build --packages-select asv_ai`
3. Source del workspace

---

## 🎬 ORDEN DE EJECUCIÓN RECOMENDADO

### Paso 1: Preparación (SIN cambios de código)
1. Crear branch nuevo: `git checkout -b generalize-package-names`
2. Hacer backup: `cp -r asv_ai asv_ai_backup`
3. Confirmar que todo compila actualmente

### Paso 2: Renombrados Estructurales (filesystem)
1. Renombrar carpeta principal `asv_ai/` → `rl_multi_agent/`
2. Renombrar subcarpetas dentro de `rl_multi_agent/rl_multi_agent/`
3. Renombrar archivos Python específicos
4. Renombrar archivo RViz

### Paso 3: Actualizar Referencias (código)
1. Actualizar `setup.py` (package_name, entry_points)
2. Actualizar `package.xml`
3. Actualizar todos los imports en archivos Python
4. Actualizar launch files

### Paso 4: Renombrar Clases y Funciones
1. Actualizar nombres de clases
2. Actualizar nombres de métodos
3. Actualizar variables internas

### Paso 5: Limpiar Strings y Documentación
1. Actualizar mensajes de log
2. Actualizar comentarios
3. Actualizar topics ROS2
4. Actualizar TF frames

### Paso 6: Verificación
1. `colcon build --packages-select rl_multi_agent`
2. Source workspace
3. Ejecutar tests
4. Lanzar sistema completo
5. Commit si todo funciona

---

## 🤔 DECISIONES PENDIENTES

### Pregunta 1: Nombre del Paquete Principal
**Opciones:**
- `rl_multi_agent` ✅ (Recomendado - enfatiza RL multi-agente)
- `ppo_multi_robot` (Más específico al algoritmo)
- `multi_robot_rl` (Enfatiza múltiples robots)
- `marl_framework` (Multi-Agent RL Framework - más académico)

**Respuesta**: rl_multi_agent

### Pregunta 2: Nombre para Robot Agent Class
**Opciones:**
- `PhysicalRobotModel` ✅ (Recomendado - describe su propósito)
- `RobotDynamics` (Enfatiza la dinámica)
- `RobotSimulator` (Enfatiza simulación)
- `RobotAgent` (Más simple pero menos descriptivo)

**Respuesta**: PhysicalRobotModel

### Pregunta 3: Valores Físicos Hardcoded
**¿Parametrizar o mantener como ejemplo?**
- Los valores en `data_conversion.py` son específicos del ASV
- ¿Crear un archivo de configuración `robot_params.yaml`?
- ¿O dejarlos como valores por defecto con comentarios?

**Respuesta**: Vamos a dejarlo como está por ahora

### Pregunta 4: Comentarios en Español
**¿Traducir a inglés todos los comentarios?**
- Actualmente hay mezcla de español e inglés
- Beneficio: Mayor accesibilidad internacional
- Costo: Tiempo adicional de revisión

**Respuesta**: Vale, traducelos
---

## 📝 NOTAS FINALES

Este plan es **completo pero reversible**. Cada fase puede ejecutarse de manera independiente y puede hacerse rollback si es necesario. 

**Recomendación:** Implementar por fases, hacer commit después de cada fase exitosa, y probar compilación/ejecución después de cada fase mayor.

**Tiempo estimado:** 
- Renombrados automáticos: 1-2 horas
- Verificaciones y ajustes: 2-3 horas
- Testing completo: 1-2 horas
- **Total: 4-7 horas de trabajo**

¿Proceder con la implementación? ¿Alguna preferencia sobre las decisiones pendientes?

---

## 📝 DIARIO DE IMPLEMENTACIÓN

### Fecha: 10 Noviembre 2025

#### ✅ PASO 1: Preparación
- [ ] 1.1 Crear backup de asv_ai/
- [ ] 1.2 Verificar compilación actual
- [ ] 1.3 Estado inicial confirmado

#### 🔄 PASO 2: Renombrados Estructurales (en progreso)
- [ ] 2.1 Renombrar carpeta principal: asv_ai/ → rl_multi_agent/
- [ ] 2.2 Renombrar subcarpetas
- [ ] 2.3 Renombrar archivos Python
- [ ] 2.4 Renombrar archivo RViz

#### 📝 PASO 3: Actualizar Referencias
- [ ] 3.1 setup.py
- [ ] 3.2 package.xml
- [ ] 3.3 Imports en archivos Python
- [ ] 3.4 Launch files

#### 🐍 PASO 4: Renombrar Clases y Funciones
- [ ] 4.1 Clases principales
- [ ] 4.2 Métodos internos
- [ ] 4.3 Variables de instancia

#### 🌐 PASO 5: Strings y Documentación
- [ ] 5.1 Mensajes de log
- [ ] 5.2 Comentarios (español → inglés)
- [ ] 5.3 Topics ROS2
- [ ] 5.4 TF frames

#### ✅ PASO 6: Verificación Final
- [ ] 6.1 Compilación exitosa
- [ ] 6.2 Tests pasando
- [ ] 6.3 Sistema funcional

---

### Cambios Realizados:

#### Inicio de Implementación (10 Nov 2025)
**Estado:** Iniciando Paso 1 - Preparación
