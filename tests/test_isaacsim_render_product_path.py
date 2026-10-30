"""CPU regression for the HydraTexture-versus-path cleanup failure."""

import ast
from pathlib import Path
from types import SimpleNamespace


def test_cleanup_uses_string_path_and_destroys_owned_texture():
    source = Path(__file__).resolve().parents[1] / "src/holosoma/holosoma/simulator/isaacsim/video_recorder.py"
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_cleanup_backend_resources"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    events = []

    class Annotator:
        def detach(self, products):
            assert products == ["/Render/HydraTextures/Test"]
            events.append("detach")

    class HydraTexture:
        path = "/Render/HydraTextures/Test"

        def destroy(self):
            events.append("destroy")

    owner = SimpleNamespace(_rgb_annotator=Annotator(), _render_product=HydraTexture(), _camera_prim_path=None)
    assert namespace["_cleanup_backend_resources"](owner) == []
    assert events == ["detach", "destroy"]
    assert owner._rgb_annotator is None and owner._render_product is None
