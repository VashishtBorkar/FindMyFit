from pathlib import Path


def test_python_sources_do_not_import_through_src_namespace():
    roots = [Path("src"), Path("backend"), Path("scripts"), Path("training"), Path("examples")]
    offenders = []
    for root in roots:
        for source in root.rglob("*.py"):
            text = source.read_text(encoding="utf-8")
            if "from src." in text or "import src." in text:
                offenders.append(str(source))
    assert offenders == []
