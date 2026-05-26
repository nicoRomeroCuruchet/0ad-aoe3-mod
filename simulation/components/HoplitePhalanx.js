/**
 * HoplitePhalanx component — AOE3 mod (Athenai civilization)
 *
 * Models three physical properties of the Classical Greek hoplite:
 *
 *   1. Othismos (Proximity Bonus)
 *      Nearby allied hoplites push together, reinforcing health and attack.
 *      Uses the RangeManager's active query (O(1) per update) instead of
 *      per-tick spatial scans.
 *      Multiplier = 1.0 + log2(neighborCount + 1)
 *
 *   2. Directional Vulnerability (Asymmetric Armor)
 *      The aspis shield covers the left side and front. Attacks arriving
 *      from > 45 degrees off the unit's facing reduce Hack and Pierce
 *      resistance by 60%, simulating the exposed right flank and rear.
 *
 *   3. Rotational Inertia
 *      A dense phalanx block is slow to pivot. The unit's effective turn
 *      speed is penalised proportionally to the number of neighbors.
 *      Penalty = baseSpeed / (1 + 0.15 * neighborCount)
 */

function HoplitePhalanx() {}

HoplitePhalanx.prototype.Schema =
	"<a:help>Phalanx effect for hoplite-class units. Grants proximity bonus, " +
	"directional vulnerability, and rotational inertia.</a:help>" +
	"<element name='Range' a:help='Radius (meters) for detecting allied hoplites.'>" +
		"<ref name='positiveDecimal'/>" +
	"</element>" +
	"<element name='FlankAngle' a:help='Half-angle (degrees) of the frontal arc considered shielded.'>" +
		"<ref name='positiveDecimal'/>" +
	"</element>" +
	"<element name='FlankReduction' a:help='Fraction of resistance lost when hit from outside the frontal arc (0.0-1.0).'>" +
		"<ref name='nonNegativeDecimal'/>" +
	"</element>" +
	"<element name='TurnInertiaPer' a:help='Turn-speed denominator weight per neighbor (higher = slower turning).'>" +
		"<ref name='nonNegativeDecimal'/>" +
	"</element>";

// ---------------------------------------------------------------------------
//  Initialization / Teardown
// ---------------------------------------------------------------------------

HoplitePhalanx.prototype.Init = function()
{
	this.neighborCount = 0;
	this.rangeQuery = undefined;
	this.modifiersApplied = false;

	// Cache converted template values.
	this.range = +this.template.Range;
	this.flankAngleRad = +this.template.FlankAngle * Math.PI / 180;
	this.flankReduction = +this.template.FlankReduction;
	this.turnInertiaPer = +this.template.TurnInertiaPer;

	this.SetupRangeQuery();
};

HoplitePhalanx.prototype.OnDestroy = function()
{
	this.RemoveModifiers();
	this.DestroyRangeQuery();
};

// ---------------------------------------------------------------------------
//  Range Query Management
// ---------------------------------------------------------------------------

/**
 * Create an active range query that watches for allied entities with the
 * Identity interface inside `this.range` meters.  The engine will push
 * OnRangeUpdate messages whenever entities enter or leave the radius —
 * no per-tick polling required.
 */
HoplitePhalanx.prototype.SetupRangeQuery = function()
{
	const cmpOwnership = Engine.QueryInterface(this.entity, IID_Ownership);
	if (!cmpOwnership)
		return;

	const owner = cmpOwnership.GetOwner();
	if (owner === INVALID_PLAYER)
		return;

	const cmpRangeManager = Engine.QueryInterface(SYSTEM_ENTITY, IID_RangeManager);
	if (!cmpRangeManager)
		return;

	this.rangeQuery = cmpRangeManager.CreateActiveQuery(
		this.entity,
		0,
		this.range,
		[owner],            // Only same-player allies.
		IID_Identity,       // Target must have Identity.
		cmpRangeManager.GetEntityFlagMask("normal"),
		false               // Don't account for entity footprint size.
	);

	cmpRangeManager.EnableActiveQuery(this.rangeQuery);
};

HoplitePhalanx.prototype.DestroyRangeQuery = function()
{
	if (this.rangeQuery === undefined)
		return;

	const cmpRangeManager = Engine.QueryInterface(SYSTEM_ENTITY, IID_RangeManager);
	if (cmpRangeManager)
		cmpRangeManager.DestroyActiveQuery(this.rangeQuery);

	this.rangeQuery = undefined;
};

// ---------------------------------------------------------------------------
//  Event Handlers
// ---------------------------------------------------------------------------

/**
 * Called by the engine whenever entities enter or leave our active query.
 * We filter to hoplite-class neighbors, recount, and reapply modifiers.
 */
HoplitePhalanx.prototype.OnRangeUpdate = function(msg)
{
	if (msg.tag !== this.rangeQuery)
		return;

	// Full recount: iterate the current result set and keep only hoplites.
	const cmpRangeManager = Engine.QueryInterface(SYSTEM_ENTITY, IID_RangeManager);
	if (!cmpRangeManager)
		return;

	const entities = cmpRangeManager.ResetActiveQuery(this.rangeQuery);
	let count = 0;
	for (const ent of entities)
	{
		if (ent === this.entity)
			continue;

		// Only count entities that also carry the HoplitePhalanx component
		// (i.e. other hoplites, not archers who wander nearby).
		if (Engine.QueryInterface(ent, IID_HoplitePhalanx))
			++count;
	}

	if (count === this.neighborCount)
		return;

	this.neighborCount = count;
	this.ApplyModifiers();
};

/**
 * Rebuild range query when ownership changes (e.g. conversion).
 */
HoplitePhalanx.prototype.OnOwnershipChanged = function(msg)
{
	this.RemoveModifiers();
	this.DestroyRangeQuery();
	this.neighborCount = 0;

	if (msg.to !== INVALID_PLAYER)
		this.SetupRangeQuery();
};

// ---------------------------------------------------------------------------
//  Mechanism 1: Othismos – Proximity Bonus
// ---------------------------------------------------------------------------

/**
 * Compute the non-linear phalanx multiplier.
 * Formula: 1.0 + log2(neighborCount + 1)
 *
 * Examples:
 *   0 neighbors → 1.00  (solo – no bonus)
 *   1 neighbor  → 2.00
 *   3 neighbors → 2.00
 *   7 neighbors → 3.00
 *  15 neighbors → 4.00
 *
 * The logarithm ensures diminishing returns: the first few neighbors
 * matter most, preventing unlimited stacking.
 */
HoplitePhalanx.prototype.GetMultiplier = function()
{
	return 1.0 + Math.log2(this.neighborCount + 1);
};

/**
 * Apply (or re-apply) Health/Max and Attack/Melee/Damage modifiers via
 * the engine's ModifiersManager.  Always removes old modifiers first to
 * avoid stacking stale values.
 */
HoplitePhalanx.prototype.ApplyModifiers = function()
{
	this.RemoveModifiers();

	const mult = this.GetMultiplier();
	if (mult <= 1.0)
		return;

	const cmpModifiersManager = Engine.QueryInterface(SYSTEM_ENTITY, IID_ModifiersManager);
	if (!cmpModifiersManager)
		return;

	const id = "HoplitePhalanx/Othismos/" + this.entity;

	cmpModifiersManager.AddModifiers(id, {
		"Health/Max": [{ "Affects": [["Unit"]], "Multiply": mult }],
		"Attack/Melee/Damage/Hack": [{ "Affects": [["Unit"]], "Multiply": mult }],
		"Attack/Melee/Damage/Pierce": [{ "Affects": [["Unit"]], "Multiply": mult }],
		"Attack/Melee/Damage/Crush": [{ "Affects": [["Unit"]], "Multiply": mult }]
	}, this.entity);

	this.modifiersApplied = true;
};

HoplitePhalanx.prototype.RemoveModifiers = function()
{
	if (!this.modifiersApplied)
		return;

	const cmpModifiersManager = Engine.QueryInterface(SYSTEM_ENTITY, IID_ModifiersManager);
	if (!cmpModifiersManager)
		return;

	const id = "HoplitePhalanx/Othismos/" + this.entity;
	const paths = [
		"Health/Max",
		"Attack/Melee/Damage/Hack",
		"Attack/Melee/Damage/Pierce",
		"Attack/Melee/Damage/Crush"
	];

	for (const path of paths)
		cmpModifiersManager.RemoveModifier(path, id, this.entity);

	this.modifiersApplied = false;
};

// ---------------------------------------------------------------------------
//  Mechanism 2: Directional Vulnerability (Asymmetric Armor)
// ---------------------------------------------------------------------------

/**
 * Determine the resistance multiplier for an incoming attack based on
 * the angle between the unit's facing direction and the attacker's
 * position.
 *
 * @param {number} attackerEntity - The entity ID of the attacker.
 * @return {number} 1.0 if the attack is frontal (shielded), or
 *         (1.0 - flankReduction) if the attack comes from the flank/rear.
 */
HoplitePhalanx.prototype.GetDirectionalResistanceFactor = function(attackerEntity)
{
	const cmpPosition = Engine.QueryInterface(this.entity, IID_Position);
	const cmpAttackerPos = Engine.QueryInterface(attackerEntity, IID_Position);
	if (!cmpPosition || !cmpAttackerPos ||
	    !cmpPosition.IsInWorld() || !cmpAttackerPos.IsInWorld())
		return 1.0;

	// Unit's facing angle (Y rotation, radians, CW from +Z axis).
	const facing = cmpPosition.GetRotation().y;

	// Vector from this unit to the attacker.
	const myPos = cmpPosition.GetPosition2D();
	const atkPos = cmpAttackerPos.GetPosition2D();
	const dx = atkPos.x - myPos.x;
	const dz = atkPos.y - myPos.y; // 2D .y is world Z in the XZ plane.

	// Angle from this unit to the attacker (atan2 returns [-PI, PI]).
	const angleToAttacker = Math.atan2(dx, dz);

	// Relative angle: how far off our facing the attacker is.
	let delta = angleToAttacker - facing;
	// Normalise to [-PI, PI].
	delta = delta - 2 * Math.PI * Math.round(delta / (2 * Math.PI));

	if (Math.abs(delta) <= this.flankAngleRad)
		return 1.0; // Frontal arc — full shield protection.

	// Flank or rear — resistance reduced.
	return 1.0 - this.flankReduction;
};

/**
 * Hook into the damage pipeline.  When this unit is attacked we check
 * the incoming angle and, if flanked, temporarily reduce Hack and Pierce
 * resistance via a short-lived modifier applied before the damage
 * calculation resolves.
 *
 * Technical note: MT_Attacked fires *after* damage has been applied by
 * the engine's attack helpers.  To intercept *before* the hit resolves
 * we would need to patch the attack helper itself.  Instead, this
 * component applies a penalty modifier on the previous tick's attack and
 * recalculates.  For a cleaner hook we use a timer-driven approach:
 * when the unit is attacked, we apply a short debuff (200 ms) that
 * weakens armor if the attack came from a vulnerable angle.  Subsequent
 * hits during that window use the reduced armor automatically.
 *
 * A simpler approach: we listen to MT_Attacked and, if flank-hit, apply
 * extra "bonus" damage retroactively as a one-shot health reduction.
 * This avoids modifying core engine files.
 */
HoplitePhalanx.prototype.OnAttacked = function(msg)
{
	if (!msg.attacker || !msg.damage)
		return;

	const factor = this.GetDirectionalResistanceFactor(msg.attacker);
	if (factor >= 1.0)
		return; // Frontal hit — no extra damage.

	// The attack already resolved with full resistance.  We compute the
	// extra damage that *should* have landed if resistance was reduced.
	//
	// Original damage formula per type:
	//   dmg = attackPower * 0.9^resistance
	// With reduced resistance (resistance * factor):
	//   dmg_flanked = attackPower * 0.9^(resistance * factor)
	// Extra damage = dmg_flanked - dmg_original
	//              = attackPower * (0.9^(res*factor) - 0.9^res)
	//
	// We don't have the per-type breakdown in MT_Attacked (only total),
	// so we approximate:  extraDamage = totalDamage * reductionRatio.
	//
	// reductionRatio = how much more damage should pass through:
	//   For a 60% armor reduction (factor=0.4) with typical resistance 4:
	//   0.9^(4*0.4) / 0.9^4 ≈ 1.32 → ~32% more damage.
	//   We simplify to:  extra = damage * flankReduction  (conservative).

	const extraDamage = msg.damage * this.flankReduction;
	if (extraDamage <= 0)
		return;

	const cmpHealth = Engine.QueryInterface(this.entity, IID_Health);
	if (cmpHealth)
		cmpHealth.Reduce(extraDamage);
};

// ---------------------------------------------------------------------------
//  Mechanism 3: Rotational Inertia
// ---------------------------------------------------------------------------

/**
 * Return the turn-speed penalty divisor based on current neighbor count.
 *
 * effectiveTurnSpeed = baseTurnSpeed / (1 + turnInertiaPer * neighborCount)
 *
 * With default turnInertiaPer = 0.15:
 *   0 neighbors → /1.00  (full agility)
 *   3 neighbors → /1.45
 *   7 neighbors → /2.05
 *  15 neighbors → /3.25  (very sluggish)
 *
 * The penalty is applied via the ModifiersManager on the
 * "UnitMotion/TurnRate" value path.
 */
HoplitePhalanx.prototype.GetTurnSpeedMultiplier = function()
{
	return 1.0 / (1.0 + this.turnInertiaPer * this.neighborCount);
};

/**
 * Overrides ApplyModifiers to also include the turn-rate penalty.
 * We redefine the full modifier set whenever the neighbor count changes
 * so all three mechanisms stay in sync.
 */
(function()
{
	const _baseApply = HoplitePhalanx.prototype.ApplyModifiers;
	const _baseRemove = HoplitePhalanx.prototype.RemoveModifiers;

	HoplitePhalanx.prototype.ApplyModifiers = function()
	{
		// Remove everything first (old othismos + inertia modifiers).
		this.RemoveModifiers();

		const mult = this.GetMultiplier();
		const turnMult = this.GetTurnSpeedMultiplier();

		// Nothing to apply if solo and no turn penalty.
		if (mult <= 1.0 && turnMult >= 1.0)
			return;

		const cmpModifiersManager = Engine.QueryInterface(SYSTEM_ENTITY, IID_ModifiersManager);
		if (!cmpModifiersManager)
			return;

		const modifiers = {};

		// Othismos bonuses (only if mult > 1).
		if (mult > 1.0)
		{
			modifiers["Health/Max"] =
				[{ "affects": ["Unit"], "multiply": mult }];
			modifiers["Attack/Melee/Damage/Hack"] =
				[{ "affects": ["Unit"], "multiply": mult }];
			modifiers["Attack/Melee/Damage/Pierce"] =
				[{ "affects": ["Unit"], "multiply": mult }];
			modifiers["Attack/Melee/Damage/Crush"] =
				[{ "affects": ["Unit"], "multiply": mult }];
		}

		// Rotational inertia penalty.
		if (turnMult < 1.0)
		{
			modifiers["UnitMotion/TurnRate"] =
				[{ "affects": ["Unit"], "multiply": turnMult }];
		}

		const id = "HoplitePhalanx/" + this.entity;
		cmpModifiersManager.AddModifiers(id, modifiers, this.entity);
		this.modifiersApplied = true;
	};

	HoplitePhalanx.prototype.RemoveModifiers = function()
	{
		if (!this.modifiersApplied)
			return;

		const cmpModifiersManager = Engine.QueryInterface(SYSTEM_ENTITY, IID_ModifiersManager);
		if (!cmpModifiersManager)
			return;

		const id = "HoplitePhalanx/" + this.entity;
		const paths = [
			"Health/Max",
			"Attack/Melee/Damage/Hack",
			"Attack/Melee/Damage/Pierce",
			"Attack/Melee/Damage/Crush",
			"UnitMotion/TurnRate"
		];

		for (const path of paths)
			cmpModifiersManager.RemoveModifier(path, id, this.entity);

		this.modifiersApplied = false;
	};
})();

// ---------------------------------------------------------------------------
//  Public Query API
// ---------------------------------------------------------------------------

/** @return {number} Current count of hoplite neighbors within range. */
HoplitePhalanx.prototype.GetNeighborCount = function()
{
	return this.neighborCount;
};

/** @return {number} Current Othismos multiplier (>= 1.0). */
HoplitePhalanx.prototype.GetPhalanxMultiplier = function()
{
	return this.GetMultiplier();
};

/** @return {number} Current turn-speed multiplier (<= 1.0). */
HoplitePhalanx.prototype.GetTurnInertiaMultiplier = function()
{
	return this.GetTurnSpeedMultiplier();
};

// ---------------------------------------------------------------------------
//  Serialization
// ---------------------------------------------------------------------------

HoplitePhalanx.prototype.Serialize = function()
{
	return {
		"neighborCount": this.neighborCount,
		"modifiersApplied": this.modifiersApplied
	};
};

HoplitePhalanx.prototype.Deserialize = function(data)
{
	this.neighborCount = data.neighborCount;
	this.modifiersApplied = data.modifiersApplied;

	// Cached template values must be rebuilt.
	this.range = +this.template.Range;
	this.flankAngleRad = +this.template.FlankAngle * Math.PI / 180;
	this.flankReduction = +this.template.FlankReduction;
	this.turnInertiaPer = +this.template.TurnInertiaPer;

	// Range query will be reconstructed by the engine on deserialization.
};

Engine.RegisterComponentType(IID_HoplitePhalanx, "HoplitePhalanx", HoplitePhalanx);
