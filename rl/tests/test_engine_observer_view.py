from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from rl.gather.engine_observer import (
    ENGINE_OBSERVER_PROTOCOL,
    EngineObserverClient,
    EngineObserverProtocolError,
    EngineObserverUnavailable,
    parse_ppm,
)
from rl.gather.env import ZeroADGatherEnv


OBSERVER_PATCH = Path("engine/patches/0ad-v0.28.0-agent-observer.patch")


class FakeResponse:
    def __init__(self, payload: bytes, content_type: str = "text/plain"):
        self._payload = payload
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit: int = -1) -> bytes:
        return self._payload if limit < 0 else self._payload[:limit]


def test_parse_ppm_accepts_the_exact_binary_frame_contract():
    payload = b"P6\n2 1\n255\n" + bytes((1, 2, 3, 4, 5, 6))

    frame = parse_ppm(payload)

    assert frame.width == 2
    assert frame.height == 1
    assert frame.ppm == payload


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"P3\n1 1\n255\n\x00\x00\x00", "P6"),
        (b"P6\n0 1\n255\n", "dimensions"),
        (b"P6\n513 1\n255\n", "dimensions"),
        (b"P6\n1 1\n127\n\x00\x00\x00", "255"),
        (b"P6\n1 1\n255\n\x00\x00", "pixel data"),
    ],
)
def test_parse_ppm_rejects_malformed_or_oversized_frames(payload, message):
    with pytest.raises(EngineObserverProtocolError, match=message):
        parse_ppm(payload)


def test_client_checks_protocol_and_requests_an_owned_entity_frame():
    requests = []
    ppm = b"P6\n1 1\n255\n\x10\x20\x30"

    def opener(request, *, timeout):
        requests.append((request, timeout))
        if request.full_url.endswith("/observer/status"):
            return FakeResponse(ENGINE_OBSERVER_PROTOCOL.encode())
        return FakeResponse(ppm, "image/x-portable-pixmap")

    client = EngineObserverClient("http://127.0.0.1:6000", opener=opener)

    client.check_available()
    frame = client.capture(42)

    assert frame.ppm == ppm
    assert [request.full_url for request, _timeout in requests] == [
        "http://127.0.0.1:6000/observer/status",
        "http://127.0.0.1:6000/observe?entity=42",
    ]
    assert all(timeout == 2.0 for _request, timeout in requests)


def test_client_reports_stock_engine_404_with_build_instruction():
    def opener(request, *, timeout):
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    client = EngineObserverClient("http://127.0.0.1:6000", opener=opener)

    with pytest.raises(EngineObserverUnavailable, match="make engine-observer"):
        client.check_available()


def test_engine_patch_restores_and_presents_the_normal_view_after_readback():
    patch = OBSERVER_PATCH.read_text()

    readback = patch.index(
        "RenderFrameForReadback(", patch.index("RenderObserverFrame")
    )
    restore = patch.index("*camera = normalCamera;", readback)
    normal_render = patch.index("g_Renderer.RenderFrame(false);", restore)
    present = patch.index("GetBackendDevice()->Present();", normal_render)

    assert readback < restore < normal_render < present


def test_engine_patch_pins_observer_los_then_restores_the_viewed_player():
    patch = OBSERVER_PATCH.read_text()
    render = patch.index("RenderObserverFrame")

    save = patch.index("normalViewedPlayer = g_Game->GetViewedPlayerID()", render)
    pin = patch.index("SetViewedPlayerID(cmpOwnership->GetOwner())", save)
    observer_dirty = patch.index("GetLOSTexture().MakeDirty()", pin)
    readback = patch.index("RenderFrameForReadback(", observer_dirty)
    restore = patch.index("SetViewedPlayerID(normalViewedPlayer)", readback)
    normal_dirty = patch.index("GetLOSTexture().MakeDirty()", restore)
    normal_render = patch.index("g_Renderer.RenderFrame(false);", normal_dirty)

    assert (
        save < pin < observer_dirty < readback < restore < normal_dirty < normal_render
    )


def test_engine_patch_disables_los_smoothing_while_switching_players():
    patch = OBSERVER_PATCH.read_text()
    render = patch.index("RenderObserverFrame")

    save = patch.index("normalSmoothLOS = g_RenderingOptions.GetSmoothLOS()", render)
    disable = patch.index("g_RenderingOptions.SetSmoothLOS(false)", save)
    observer_dirty = patch.index("GetLOSTexture().MakeDirty()", disable)
    readback = patch.index("RenderFrameForReadback(", observer_dirty)
    restore = patch.index(
        "g_RenderingOptions.SetSmoothLOS(normalSmoothLOS)", readback
    )
    normal_dirty = patch.index("GetLOSTexture().MakeDirty()", restore)

    assert save < disable < observer_dirty < readback < restore < normal_dirty


def test_engine_patch_uses_a_top_down_camera_scaled_to_live_vision():
    patch = OBSERVER_PATCH.read_text()
    render = patch.index("RenderObserverFrame")
    readback = patch.index("RenderFrameForReadback(", render)
    observer_camera = patch[render:readback]

    assert "SetPerspectiveProjection(" in observer_camera
    assert "std::atan(visionRange / OBSERVER_CAMERA_HEIGHT)" in observer_camera
    assert "SetOrthoProjection(" not in observer_camera


def test_engine_patch_orients_observer_pixels_like_agent_coordinates():
    patch = OBSERVER_PATCH.read_text()
    encode_start = patch.index("std::string MakePPM")
    encode_end = patch.index("return ppm;", encode_start)
    encoder = patch[encode_start:encode_end]
    render_start = patch.index("RenderObserverFrame")
    readback = patch.index("RenderFrameForReadback(", render_start)
    observer_camera = patch[render_start:readback]

    # OpenGL supplies rows bottom-up, while the gather observation treats
    # local +X as right and local +Z as down on screen.
    assert "CVector3D(0.0f, 0.0f, -1.0f)" in observer_camera
    assert "sourceRow = OBSERVER_FRAME_SIZE - row - 1" in encoder
    assert "column < OBSERVER_FRAME_SIZE" in encoder
    assert "sourceColumn = OBSERVER_FRAME_SIZE - column - 1" in encoder
    assert (
        "(sourceRow * OBSERVER_FRAME_SIZE + sourceColumn) * 3" in encoder
    )
    assert "ppm.resize(headerSize + bottomUpRGB.size())" in encoder
    assert "targetPixel =" in encoder
    assert "ppm[targetPixel + 2]" in encoder
    assert "ppm.append(" not in encoder


def test_observer_image_response_does_not_opt_into_browser_cors():
    patch = OBSERVER_PATCH.read_text()
    observe = patch.index('else if (uri == "/observe")')
    response_start = patch.index('"HTTP/1.1 200 OK', observe)
    response_end = patch.index("mg_write", response_start)
    response = patch[response_start:response_end]

    assert "Access-Control-Allow-Origin" not in response


def test_observer_builder_skips_unneeded_debug_spidermonkey():
    builder = Path("engine/build_observer.sh").read_text()
    patch = OBSERVER_PATCH.read_text()

    assert "export BUILD_RELEASE_ONLY=1" in builder
    assert "LLVM_OBJDUMP" in builder
    assert "RUSTUP_HOME" in builder
    assert "20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c" in builder
    assert "cbindgen@0.29.0" in builder
    assert "download_verified" in builder
    assert ".observer-source-ready" in builder
    assert ".observer-build.lock" in builder
    assert 'flock --nonblock "$builder_lock_fd"' in builder
    assert '[[ "$rustc_version" != "rustc $RUST_TOOLCHAIN "' in builder
    assert '[[ "$cbindgen_version" != "cbindgen 0.29.0"' in builder
    assert builder.count("--forward") == 3
    for dependency in (
        "libenet:libenet-dev",
        "sdl2:libsdl2-dev",
        "libpng:libpng-dev",
        "libcurl:libcurl4-gnutls-dev",
        "libsodium:libsodium-dev",
        "freetype2:libfreetype-dev",
        "icu-i18n:libicu-dev",
        "libxml-2.0:libxml2-dev",
        "x11:libx11-dev",
    ):
        assert dependency in builder
    assert "fmt/printf.h" in builder
    assert "boost/random/linear_congruential.hpp" in builder
    assert "libboost-filesystem-dev" in builder
    for option in ("--without-atlas", "--without-audio", "--without-lobby"):
        assert option in builder
    assert "BUILD_RELEASE_ONLY" in patch


def test_engine_patch_defines_compiled_module_before_registry_instantiation():
    patch = OBSERVER_PATCH.read_text()
    module_loader = patch.index("diff --git a/source/scriptinterface/ModuleLoader.h")
    module_patch = patch[module_loader:]
    compiled_module = module_patch.index("+\tclass CompiledModule")
    registry = module_patch.index(" \tusing RegistryType", compiled_module)

    assert compiled_module < registry
    assert "+class ModuleLoader::CompiledModule" not in module_patch


def test_engine_patch_supports_the_host_font_and_fmt_versions():
    patch = OBSERVER_PATCH.read_text()

    assert "diff --git a/source/graphics/FontManager.h" in patch
    assert '+#include "graphics/Font.h"' in patch
    assert "-class CFont;" in patch

    added_lines = [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    assert not any("{:?}" in line for line in added_lines)
    assert sum("'{}'" in line for line in added_lines) >= 3


def test_engine_patch_omits_gloox_when_lobby_is_disabled():
    patch = OBSERVER_PATCH.read_text()
    hw_detect = patch.index("diff --git a/source/ps/GameSetup/HWDetect.cpp")
    hw_detect_patch = patch[hw_detect:]

    assert hw_detect_patch.count("+#if CONFIG2_LOBBY") >= 2
    assert hw_detect_patch.count("+#endif") >= 2


def test_engine_patch_sorts_linear_allocator_batches_without_temporary_lists():
    patch = OBSERVER_PATCH.read_text()
    text_renderer = patch.index("diff --git a/source/graphics/TextRenderer.cpp")
    text_renderer_patch = patch[text_renderer:]

    assert "-\tm_Batches.sort(SBatchCompare());" in text_renderer_patch
    assert "+\t\t\tm_Batches.splice(insertion, m_Batches, current);" in text_renderer_patch


def test_environment_captures_the_current_polites_entity():
    expected_frame = parse_ppm(b"P6\n1 1\n255\n\x00\x00\x00")

    class Villager:
        def id(self):
            return 73

    class State:
        def units(self, *, owner, entity_type):
            assert (owner, entity_type) == (1, "polites")
            return [Villager()]

    class Observer:
        def __init__(self):
            self.entities = []

        def capture(self, entity_id):
            self.entities.append(entity_id)
            return expected_frame

    observer = Observer()
    env = ZeroADGatherEnv.__new__(ZeroADGatherEnv)
    env.game = SimpleNamespace(current_state=State())
    env.engine_observer = observer

    frame = env.capture_agent_frame()

    assert frame is expected_frame
    assert observer.entities == [73]


@pytest.mark.parametrize("entity_id", [None, 0, -1, True, "7"])
def test_client_rejects_invalid_entity_ids_without_a_request(entity_id):
    client = EngineObserverClient(
        "http://127.0.0.1:6000",
        opener=lambda *_args, **_kwargs: pytest.fail("unexpected HTTP request"),
    )

    with pytest.raises(ValueError, match="positive integer"):
        client.capture(entity_id)
