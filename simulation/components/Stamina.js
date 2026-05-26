/**
 * Stamina component - AOE3 mod
 *
 * La unidad corre mas rapido cuando tiene estamina.
 * Se cansa al moverse y recupera estamina al estar quieta.
 *
 * Estados:
 *   normal    -> velocidad normal (multiplicador 1.0)
 *   corriendo -> velocidad alta (SprintMultiplier) mientras tenga estamina
 *   agotado   -> velocidad reducida (ExhaustedMultiplier) hasta recuperar ExhaustedThreshold
 */

function Stamina() {}

Stamina.prototype.Schema =
	"<element name='Max'><data type='decimal'/></element>" +
	"<element name='DrainPerSecond'><data type='decimal'/></element>" +
	"<element name='RegenPerSecond'><data type='decimal'/></element>" +
	"<element name='SprintMultiplier'><data type='decimal'/></element>" +
	"<element name='ExhaustedMultiplier'><data type='decimal'/></element>" +
	"<element name='ExhaustedThreshold'><data type='decimal'/></element>";

Stamina.prototype.Init = function()
{
	this.stamina   = +this.template.Max;
	this.exhausted = false;

	// Tick cada 500ms
	const cmpTimer = Engine.QueryInterface(SYSTEM_ENTITY, IID_Timer);
	cmpTimer.SetInterval(this.entity, IID_Stamina, "Tick", 500, 500, null);
};

Stamina.prototype.Tick = function(data, lateness)
{
	const cmpUnitMotion = Engine.QueryInterface(this.entity, IID_UnitMotion);
	if (!cmpUnitMotion)
		return;

	const dt       = (500 + lateness) / 1000; // segundos transcurridos
	const max      = ApplyValueModificationsToEntity("Stamina/Max", +this.template.Max, this.entity);
	const drain    = ApplyValueModificationsToEntity("Stamina/DrainPerSecond", +this.template.DrainPerSecond, this.entity);
	const regen    = ApplyValueModificationsToEntity("Stamina/RegenPerSecond", +this.template.RegenPerSecond, this.entity);
	const sprint   = ApplyValueModificationsToEntity("Stamina/SprintMultiplier", +this.template.SprintMultiplier, this.entity);
	const exhaMult = ApplyValueModificationsToEntity("Stamina/ExhaustedMultiplier", +this.template.ExhaustedMultiplier, this.entity);
	const exaThres = ApplyValueModificationsToEntity("Stamina/ExhaustedThreshold", +this.template.ExhaustedThreshold, this.entity);
	const isMoving = cmpUnitMotion.GetCurrentSpeed() > 0;

	if (isMoving)
	{
		// Drenar estamina al correr
		this.stamina = Math.max(0, this.stamina - drain * dt);

		if (this.stamina === 0 && !this.exhausted)
		{
			// Se agoto - penalizacion de velocidad
			this.exhausted = true;
			cmpUnitMotion.SetSpeedMultiplier(exhaMult);
		}
		else if (!this.exhausted)
		{
			// Velocidad proporcional a la estamina restante
			const ratio = this.stamina / max;
			const mult  = exhaMult + (sprint - exhaMult) * ratio;
			cmpUnitMotion.SetSpeedMultiplier(mult);
		}
	}
	else
	{
		// Recuperar estamina al estar quieto
		this.stamina = Math.min(max, this.stamina + regen * dt);

		if (this.exhausted && this.stamina >= exaThres)
		{
			// Ya recupero suficiente para volver a correr
			this.exhausted = false;
		}

		if (!this.exhausted)
		{
			// Restaurar velocidad normal
			const ratio = this.stamina / max;
			const mult  = 1.0 + (sprint - 1.0) * ratio;
			cmpUnitMotion.SetSpeedMultiplier(mult);
		}
	}
};

Stamina.prototype.GetStamina    = function() { return this.stamina; };
Stamina.prototype.GetMaxStamina = function() { return +this.template.Max; };
Stamina.prototype.IsExhausted   = function() { return this.exhausted; };

Stamina.prototype.Serialize = function()
{
	return { "stamina": this.stamina, "exhausted": this.exhausted };
};

Stamina.prototype.Deserialize = function(data)
{
	this.stamina   = data.stamina;
	this.exhausted = data.exhausted;
};

Engine.RegisterComponentType(IID_Stamina, "Stamina", Stamina);
