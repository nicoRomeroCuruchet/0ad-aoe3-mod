/**
 * aoe3 mod - unit_panel_aoe3.js
 *
 * Loaded after selection_panels.js and selection_details.js (u > s alphabetically).
 *
 * 1. Patches displaySingle to show gather rates as visible text when the
 *    aldeano is idle (not currently carrying resources or building).
 * 2. Patches Construction panel to show build time on each building button.
 * 3. Patches Training panel to always show gather rates on gatherer buttons.
 */

// ── 1. Show gather rates in selection detail area ─────────────────────────────
(function() {
	const _origDisplaySingle = displaySingle;

	displaySingle = function(entState, template)
	{
		_origDisplaySingle(entState, template);

		const ratesIcon = Engine.GetGUIObjectByName("resourceCarryingIcon");
		const ratesText = Engine.GetGUIObjectByName("resourceCarryingText");

		// Only inject when the base code left this area hidden (unit is idle)
		// and the unit actually has gather rates.
		if (!ratesIcon.hidden || !entState.resourceGatherRates)
			return;

		const rates = entState.resourceGatherRates;

		// Collect highest rate per resource type (food, wood, stone, metal).
		const resources = ["food", "wood", "stone", "metal"];
		const parts = [];
		for (const res of resources)
		{
			let best = 0;
			for (const key in rates)
				if (key.startsWith(res + ".") && rates[key] > best)
					best = rates[key];
			if (best > 0)
				parts.push(resourceIcon(res) + " " + best.toFixed(2));
		}

		if (!parts.length)
			return;

		// Show rates as two lines of two resources each so they fit in the small text area.
		const line1 = parts.slice(0, 2).join("  ");
		const line2 = parts.slice(2).join("  ");
		const caption = line2 ? line1 + "\n" + line2 : line1;

		ratesIcon.hidden = false;
		ratesText.hidden = false;

		// Use a generic "rates" indicator icon (harvest/repair icon available in base game).
		ratesIcon.sprite = "stretched:session/icons/stances/passive.png";
		ratesIcon.tooltip = getGatherTooltip(entState);

		ratesText.caption = caption;
	};
})();

// ── 1b. Always show attack & resistance stats panel (not just on hover) ──────
(function() {
	const _origDisplaySingle2 = displaySingle;

	displaySingle = function(entState, template)
	{
		_origDisplaySingle2(entState, template);

		// Force the stats panel visible with attack + resistance info
		const statsObj = Engine.GetGUIObjectByName("attackAndResistanceStats");
		if (!statsObj)
			return;

		const tips = [
			getAttackTooltip,
			getResistanceTooltip,
			getSpeedTooltip
		].map(func => func(entState)).filter(tip => tip).join("\n");

		if (tips)
		{
			statsObj.hidden = false;
			statsObj.tooltip = tips;
		}
	};
})();

// ── 2. Construction buttons: add build time to each tooltip ───────────────────
(function() {
	const _orig = g_SelectionPanels.Construction.setupButton;
	g_SelectionPanels.Construction.setupButton = function(data)
	{
		const result = _orig.call(this, data);
		if (!result)
			return result;

		const template = GetTemplateData(data.item, data.player);
		if (template && template.cost && template.cost.time)
			data.button.tooltip += "\n" + headerFont("Tiempo de construccion:") +
				" " + Math.round(template.cost.time) + " s";

		return result;
	};
})();

// ── 3. Training buttons: always show stats (gather rates + combat) ────────────
(function() {
	const _orig = g_SelectionPanels.Training.setupButton;
	g_SelectionPanels.Training.setupButton = function(data)
	{
		const result = _orig.call(this, data);
		if (!result)
			return result;

		// Detailed tooltips already include everything.
		if (Engine.ConfigDB_GetValue("user", "showdetailedtooltips") === "true")
			return result;

		const template = GetTemplateData(data.item, data.player);
		if (!template)
			return result;

		// Gather rates for workers
		const gatherTip = getGatherTooltip(template);
		if (gatherTip)
			data.button.tooltip += "\n" + gatherTip;

		// Attack stats for military units
		const attackTip = getAttackTooltip(template);
		if (attackTip)
			data.button.tooltip += "\n" + attackTip;

		// Resistance stats
		const resistTip = getResistanceTooltip(template);
		if (resistTip)
			data.button.tooltip += "\n" + resistTip;

		// Health
		const healthTip = getHealthTooltip(template);
		if (healthTip)
			data.button.tooltip += "\n" + healthTip;

		return result;
	};
})();
