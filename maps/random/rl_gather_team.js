Engine.LoadLibrary("rmgen");
Engine.LoadLibrary("rmgen-common");

// M2: a seeded rotation of six scarce trees around one shared dropsite.
export function* generateMap()
{
	const tGrass = "medit_grass_field";
	globalThis.g_Map = new RandomMap(0, tGrass);

	const hutPosition = g_Map.getCenter();
	const treeCount = 6;
	// Twelve map tiles is 48 m in this 128-tile scenario: far enough that a
	// depleted-tree switch costs time, close enough for a five-load episode.
	const treeRadius = 12;
	const treeRotation = randFloat(0, 2 * Math.PI);

	// A compact, symmetric starting formation. It stays fixed while the ring
	// rotates, so the policy must use geometry rather than memorize a tree slot.
	g_Map.placeEntityPassable("units/athenai/polites", 1, new Vector2D(hutPosition.x - 5, hutPosition.y - 5), 0);
	g_Map.placeEntityPassable("units/athenai/polites", 1, new Vector2D(hutPosition.x - 5, hutPosition.y + 5), 0);
	g_Map.placeEntityPassable("units/athenai/polites", 1, new Vector2D(hutPosition.x + 5, hutPosition.y - 5), 0);
	g_Map.placeEntityPassable("units/athenai/polites", 1, new Vector2D(hutPosition.x + 5, hutPosition.y + 5), 0);

	// One shared hut and six equal-angle, one-worker M2 trees. Four villagers
	// must clear an initial wave, then globally assign the freed workers to the
	// last two trees; a lucky one-shot matching cannot finish the benchmark.
	g_Map.placeEntityPassable("structures/athenai/rl_storehouse", 1, hutPosition, 0);
	for (let index = 0; index < treeCount; ++index)
	{
		const treeAngle = treeRotation + index * (2 * Math.PI / treeCount);
		const treePosition = Vector2D.add(hutPosition, new Vector2D(treeRadius, 0).rotate(treeAngle));
		g_Map.placeEntityPassable("gaia/tree/rl_m2_oak", 0, treePosition, 0);
	}

	yield 100;
	return g_Map;
}
