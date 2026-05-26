Engine.LoadLibrary("rmgen");
Engine.LoadLibrary("rmgen-common");

export function* generateMap(mapSettings)
{
	// ── Primary terrain textures ──────────────────────────────────
	const tGrass        = "medit_grass_field";
	const tGrassRough   = "medit_grass_wild";
	const tGrassDry     = "medit_grass_shrubs";
	const tDirt         = "medit_dirt";
	const tDirtRough    = "medit_dirt_b";
	const tCliff        = "medit_cliff_aegean";
	const tShoreUpper   = "medit_sand";
	const tShoreLower   = "medit_sand_wet";
	const tWater        = "medit_sand_wet";
	const tForestFloor  = "medit_grass_wild";

	// ── Trees ─────────────────────────────────────────────────────
	const oOak         = "gaia/tree/oak";
	const oCypress     = "gaia/tree/cypress";
	const oPoplar      = "gaia/tree/poplar_lombardy";
	const oPine        = "gaia/tree/pine_maritime_short";
	const oDatePalm    = "gaia/tree/cretan_date_palm_tall";
	const oCarob       = "gaia/tree/carob";

	// ── Resources ─────────────────────────────────────────────────
	const oGoldMine    = "gaia/ore/mediterranean_large";
	const oStoneMine   = "gaia/rock/mediterranean_large";
	const oBerries     = "gaia/fruit/berry_01";
	const oGrapes      = "gaia/fruit/grapes";
	const oDeer        = "gaia/fauna_deer";
	const oRabbit      = "gaia/fauna_rabbit";
	const oFish        = "gaia/fish/generic";

	// ── Decorative actors ─────────────────────────────────────────
	const aRockLarge   = "actor|geology/stone_granite_med.xml";
	const aRockMedium  = "actor|geology/stone_granite_small.xml";
	const aGrassLarge  = "actor|props/flora/grass_soft_large_tall.xml";
	const aGrassShort  = "actor|props/flora/grass_soft_large.xml";
	const aBushMed     = "actor|props/flora/bush_medit_me.xml";
	const aBushSmall   = "actor|props/flora/bush_medit_sm.xml";
	const aReeds       = "actor|props/flora/cattails.xml";
	const aLillies     = "actor|props/flora/water_lillies.xml";

	// ── 1. Initialize map ─────────────────────────────────────────
	globalThis.g_Map = new RandomMap(5, tGrass);

	const mapCenter = g_Map.getCenter();
	const mapSize   = g_Map.getSize();
	const mapRadius = mapSize / 2;

	// ── 2. Tile classes ───────────────────────────────────────────
	const clPlayer = g_Map.createTileClass();
	const clForest = g_Map.createTileClass();
	const clMetal  = g_Map.createTileClass();
	const clStone  = g_Map.createTileClass();
	const clFood   = g_Map.createTileClass();
	const clWater  = g_Map.createTileClass();
	const clHill   = g_Map.createTileClass();
	const clDirt   = g_Map.createTileClass();
	const clShore  = g_Map.createTileClass();

	// ── 3. Natural lake with smooth shoreline ─────────────────────
	// Irregular lake shape offset from center for asymmetry
	const lakeCenter = new Vector2D(
		mapCenter.x + randFloat(-mapRadius * 0.08, mapRadius * 0.08),
		mapCenter.y + randFloat(-mapRadius * 0.08, mapRadius * 0.08)
	);

	// Deep water core
	createArea(
		new ClumpPlacer(diskArea(mapRadius * 0.15), 0.7, 0.08, Infinity, lakeCenter),
		[
			new TerrainPainter(tWater),
			new SmoothElevationPainter(ELEVATION_SET, -4, 3),
			new TileClassPainter(clWater)
		]
	);

	// Shallow shore ring around the lake
	createArea(
		new ClumpPlacer(diskArea(mapRadius * 0.22), 0.7, 0.08, Infinity, lakeCenter),
		[
			new TerrainPainter(tShoreLower),
			new SmoothElevationPainter(ELEVATION_SET, -1, 4),
			new TileClassPainter(clWater)
		],
		new NullConstraint()
	);

	// Sandy shore border
	createArea(
		new ClumpPlacer(diskArea(mapRadius * 0.27), 0.7, 0.08, Infinity, lakeCenter),
		[
			new TerrainPainter(tShoreUpper),
			new TileClassPainter(clShore)
		],
		avoidClasses(clWater, 0)
	);
	yield 10;

	// ── 4. Elevation: bumps across the map ────────────────────────
	createBumps(
		avoidClasses(clWater, 3, clPlayer, 5),
		scaleByMapSize(80, 180),
		1, 5, 3, 0, 2
	);
	yield 15;

	// ── 5. Hills ──────────────────────────────────────────────────
	createHills(
		[tCliff, tCliff, tGrassRough],
		avoidClasses(clPlayer, 20, clWater, 8, clHill, 15),
		clHill,
		scaleByMapSize(2, 6),
		1, 4, 16, 0.5, 12, 2
	);
	yield 20;

	// ── 6. Player bases ───────────────────────────────────────────
	const { playerIDs, playerPosition } = playerPlacementCircle(fractionToTiles(0.35));

	for (let i = 0; i < getNumPlayers(); ++i)
	{
		const pos = playerPosition[i];

		placeCivDefaultStartingEntities(pos, playerIDs[i], false);

		// Flatten and mark player area
		createArea(
			new ClumpPlacer(diskArea(15), 0.9, 0.5, Infinity, pos),
			[
				new TerrainPainter(tGrass),
				new SmoothElevationPainter(ELEVATION_SET, 3, 4),
				new TileClassPainter(clPlayer)
			]
		);

		// Nearby berry patch per player
		const berryAngle = randFloat(0, 2 * Math.PI);
		const berryPos = Vector2D.add(pos, new Vector2D(14, 0).rotate(berryAngle));
		const berryGroup = new SimpleGroup(
			[new SimpleObject(oBerries, 5, 6, 1, 2)],
			true, clFood
		);
		berryGroup.setCenterPosition(berryPos);
		createObjectGroup(berryGroup, 0);

		// Nearby trees per player
		const treeAngle = berryAngle + Math.PI * 0.6;
		const treePos = Vector2D.add(pos, new Vector2D(16, 0).rotate(treeAngle));
		const treeGroup = new SimpleGroup(
			[new SimpleObject(oOak, 4, 6, 1, 3)],
			true, clForest
		);
		treeGroup.setCenterPosition(treePos);
		createObjectGroup(treeGroup, 0);
	}
	yield 35;

	// ── 7. Forests (mixed species) ────────────────────────────────
	const forestTypes = [
		[tForestFloor, tForestFloor, tForestFloor,
		 tForestFloor + TERRAIN_SEPARATOR + oOak,
		 tForestFloor + TERRAIN_SEPARATOR + oCypress],
		[tForestFloor, tForestFloor, tForestFloor,
		 tForestFloor + TERRAIN_SEPARATOR + oPine,
		 tForestFloor + TERRAIN_SEPARATOR + oPoplar],
		[tForestFloor, tForestFloor, tForestFloor,
		 tForestFloor + TERRAIN_SEPARATOR + oCarob,
		 tForestFloor + TERRAIN_SEPARATOR + oDatePalm]
	];

	for (const fType of forestTypes)
	{
		createForests(
			fType,
			avoidClasses(clPlayer, 18, clForest, 12, clWater, 5, clHill, 3),
			clForest,
			scaleByMapSize(30, 60)
		);
	}
	yield 50;

	// ── 8. Terrain patches (dirt, dry grass) ──────────────────────
	createLayeredPatches(
		[scaleByMapSize(3, 6), scaleByMapSize(5, 10), scaleByMapSize(8, 21)],
		[[tGrassRough, tGrassDry], [tGrassDry, tDirt], [tDirt, tDirtRough]],
		[1, 1],
		avoidClasses(clWater, 3, clForest, 2, clPlayer, 8, clDirt, 5, clHill, 2),
		scaleByMapSize(15, 45),
		clDirt
	);
	yield 55;

	// ── 9. Gold mines ─────────────────────────────────────────────
	createObjectGroupsDeprecated(
		new SimpleGroup([new SimpleObject(oGoldMine, 1, 1, 0, 0)], true, clMetal),
		0,
		avoidClasses(clPlayer, 25, clMetal, 20, clWater, 5, clForest, 4, clHill, 3),
		scaleByMapSize(6, 14), 100
	);
	yield 62;

	// ── 10. Stone mines ───────────────────────────────────────────
	createObjectGroupsDeprecated(
		new SimpleGroup([new SimpleObject(oStoneMine, 1, 1, 0, 0)], true, clStone),
		0,
		avoidClasses(clPlayer, 25, clMetal, 10, clStone, 20, clWater, 5, clForest, 4, clHill, 3),
		scaleByMapSize(6, 14), 100
	);
	yield 68;

	// ── 11. Berry bushes ──────────────────────────────────────────
	createObjectGroupsDeprecated(
		new SimpleGroup([new SimpleObject(oBerries, 4, 7, 1, 3)], true, clFood),
		0,
		avoidClasses(clPlayer, 14, clFood, 15, clWater, 4, clForest, 3),
		scaleByMapSize(8, 16), 100
	);
	yield 72;

	// ── 12. Grape vines ───────────────────────────────────────────
	createObjectGroupsDeprecated(
		new SimpleGroup([new SimpleObject(oGrapes, 3, 5, 1, 2)], true, clFood),
		0,
		avoidClasses(clPlayer, 14, clFood, 12, clWater, 4, clForest, 3),
		scaleByMapSize(4, 8), 100
	);
	yield 75;

	// ── 13. Deer herds ────────────────────────────────────────────
	createObjectGroupsDeprecated(
		new SimpleGroup([new SimpleObject(oDeer, 3, 6, 2, 4)], true, clFood),
		0,
		avoidClasses(clPlayer, 14, clFood, 10, clWater, 4),
		scaleByMapSize(6, 12), 100
	);
	yield 78;

	// ── 14. Rabbits ───────────────────────────────────────────────
	createObjectGroupsDeprecated(
		new SimpleGroup([new SimpleObject(oRabbit, 2, 4, 1, 3)], true, clFood),
		0,
		avoidClasses(clPlayer, 10, clFood, 8, clWater, 3),
		scaleByMapSize(4, 10), 100
	);
	yield 80;

	// ── 15. Fish in the lake ──────────────────────────────────────
	createObjectGroupsDeprecated(
		new SimpleGroup([new SimpleObject(oFish, 1, 2, 0, 2)], true, clFood),
		0,
		[stayClasses(clWater, 4), avoidClasses(clFood, 10)],
		scaleByMapSize(6, 14), 50
	);
	yield 83;

	// ── 16. Straggler trees (scattered individuals) ───────────────
	const stragglerTrees = [oOak, oCypress, oPoplar, oCarob, oDatePalm];
	for (const tree of stragglerTrees)
	{
		createObjectGroupsDeprecated(
			new SimpleGroup([new SimpleObject(tree, 1, 1, 0, 1)], true, clForest),
			0,
			avoidClasses(clPlayer, 10, clForest, 4, clWater, 4, clHill, 2),
			scaleByMapSize(4, 12), 50
		);
	}
	yield 88;

	// ── 17. Decorative rocks ──────────────────────────────────────
	createObjectGroupsDeprecated(
		new SimpleGroup(
			[new SimpleObject(aRockMedium, 1, 3, 0, 1),
			 new SimpleObject(aRockLarge, 0, 1, 0, 1)],
			true
		),
		0,
		avoidClasses(clPlayer, 8, clForest, 2, clWater, 2),
		scaleByMapSize(16, 60), 50
	);
	yield 91;

	// ── 18. Decorative grass and bushes ───────────────────────────
	createObjectGroupsDeprecated(
		new SimpleGroup(
			[new SimpleObject(aGrassLarge, 1, 2, 0, 1),
			 new SimpleObject(aGrassShort, 2, 4, 0, 2)],
			true
		),
		0,
		avoidClasses(clPlayer, 6, clWater, 3, clForest, 1, clHill, 2),
		scaleByMapSize(30, 120), 50
	);

	createObjectGroupsDeprecated(
		new SimpleGroup(
			[new SimpleObject(aBushMed, 1, 2, 0, 2),
			 new SimpleObject(aBushSmall, 1, 3, 0, 2)],
			true
		),
		0,
		avoidClasses(clPlayer, 6, clWater, 3, clForest, 1),
		scaleByMapSize(20, 80), 50
	);
	yield 94;

	// ── 19. Reeds and water lillies along the shore ───────────────
	createObjectGroupsDeprecated(
		new SimpleGroup([new SimpleObject(aReeds, 2, 5, 0, 2)], true),
		0,
		[stayClasses(clShore, 0), avoidClasses(clForest, 1)],
		scaleByMapSize(10, 30), 50
	);

	createObjectGroupsDeprecated(
		new SimpleGroup([new SimpleObject(aLillies, 1, 3, 0, 2)], true),
		0,
		[stayClasses(clWater, 1), avoidClasses(clFood, 4)],
		scaleByMapSize(6, 16), 50
	);
	yield 97;

	// ── 20. Set environment ───────────────────────────────────────
	setSkySet("sunny");
	setSunColor(0.95, 0.92, 0.78);
	setWaterColor(0.02, 0.17, 0.52);
	setWaterTint(0.3, 0.45, 0.65);
	setWaterWaviness(3);
	setWaterMurkiness(0.7);

	setFogFactor(0.2);
	setFogThickness(0.15);

	setPPEffect("hdr");
	setPPSaturation(0.55);
	setPPBloom(0.3);

	yield 100;

	return g_Map;
}
