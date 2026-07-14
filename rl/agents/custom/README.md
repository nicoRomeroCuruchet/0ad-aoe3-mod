# Implementar un agente propio

Cada algoritmo se divide en dos objetos chicos:

```python
class MyPolicy:
    def act(self, observation, *, deterministic):
        """Devuelve una acción numpy compatible con env.action_space."""


class MyTrainer:
    def fit(self, request):
        # request.env, request.total_steps, request.seed y request.agent.parameters
        # están disponibles acá.
        return MyPolicy(...)
```

`Policy` es lo único que conoce el evaluador. `Trainer` es dueño del loop de
aprendizaje, por lo que puede ser SB3, PyTorch puro, JAX o una implementación
tabular sin cambiar el entorno ni la evaluación.

Para conectar un algoritmo:

1. Creá un módulo en esta carpeta e implementá los protocolos de
   `rl.agents.base`.
2. Agregá una entrada explícita a `_AGENTS` en `rl/agents/registry.py` con las
   capacidades que soporte: construir, entrenar, cargar y/o guardar.
   `rl.train` exige `trainer_factory` y `saver` para no terminar una corrida
   larga sin poder guardar el resultado.
3. Creá un TOML en `rl/configs/` que use ese nombre y declare sólo sus
   hiperparámetros dentro de `[agent]`.
4. Agregá tests de contrato en `rl/tests/agents/`. El test no debe arrancar
   0 A.D.; inyectá un entorno mínimo o un backend falso.

No pongas lógica del algoritmo en `train.py`, `eval.py` ni `gather/env.py`.
