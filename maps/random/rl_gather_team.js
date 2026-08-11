Engine.LoadLibrary("rmgen");
Engine.LoadLibrary("rmgen-common");

// Mapa minimo y DETERMINISTA para M2: 4 aldeanos + 4 arboles + 1 deposito, sin RNG.
export function* generateMap()
{
	const tGrass = "medit_grass_field";
	globalThis.g_Map = new RandomMap(0, tGrass);

	const c = g_Map.getCenter();

	// Cuatro aldeanos separados, a la izquierda del centro.
	g_Map.placeEntityPassable("units/athenai/polites", 1, new Vector2D(c.x - 20, c.y - 12), 0);
	g_Map.placeEntityPassable("units/athenai/polites", 1, new Vector2D(c.x - 20, c.y - 4), 0);
	g_Map.placeEntityPassable("units/athenai/polites", 1, new Vector2D(c.x - 20, c.y + 4), 0);
	g_Map.placeEntityPassable("units/athenai/polites", 1, new Vector2D(c.x - 20, c.y + 12), 0);

	// Un unico deposito: obliga a volver al mismo punto con la carga.
	g_Map.placeEntityPassable("structures/athenai/rl_storehouse", 1, new Vector2D(c.x - 28, c.y + 10), 0);

	// Cuatro arboles separados entre si: existe una asignacion perfecta.
	g_Map.placeEntityPassable("gaia/tree/oak", 0, new Vector2D(c.x + 20, c.y - 18), 0);
	g_Map.placeEntityPassable("gaia/tree/oak", 0, new Vector2D(c.x + 20, c.y - 6), 0);
	g_Map.placeEntityPassable("gaia/tree/oak", 0, new Vector2D(c.x + 20, c.y + 6), 0);
	g_Map.placeEntityPassable("gaia/tree/oak", 0, new Vector2D(c.x + 20, c.y + 18), 0);

	yield 100;
	return g_Map;
}
