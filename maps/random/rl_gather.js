Engine.LoadLibrary("rmgen");
Engine.LoadLibrary("rmgen-common");

// Mapa mínimo y DETERMINISTA para RL: 1 aldeano + 1 árbol, sin RNG.
function* GenerateMap()
{
	const tGrass = "medit_grass_field";
	globalThis.g_Map = new RandomMap(0, tGrass);

	const c = g_Map.getCenter();

	// Aldeano del mod (player 1), a la izquierda del centro.
	g_Map.placeEntityPassable("units/athenai/polites", 1, new Vector2D(c.x - 20, c.y), 0);

	// Recurso gaia (player 0), a la derecha del centro.
	g_Map.placeEntityPassable("gaia/tree/oak", 0, new Vector2D(c.x + 20, c.y), 0);

	yield 100;
	return g_Map;
}
