from app.agent.skills import SkillManager
from app.agent.tools.manager import ToolManager


def test_expert_catalog_and_execution_share_capability_boundary():
    skills = SkillManager()
    tools = {"load_skill", "query_product", "search_products"}
    catalog = skills.build_catalog_prompt(tools)
    assert "product-recommend" in catalog
    assert "process-return" not in catalog and "track-order" not in catalog
    manager = ToolManager(allowed_tools=tools, local_tool_overrides={"load_skill": skills.load_skill})
    import json
    result = json.loads(manager.execute_tool("load_skill", {"skill_name": "process-return"}))
    assert result["error_code"] == "SKILL_CAPABILITY_UNAVAILABLE"
    good = json.loads(manager.execute_tool("load_skill", {"skill_name": "product-recommend"}))
    assert good["success"] and len(good["version"]) == 64
    assert skills.build_catalog_prompt({"query_product", "search_products"}) == ""


def test_skill_update_does_not_silently_use_stale_cache(tmp_path):
    folder = tmp_path / "example"
    folder.mkdir()
    path = folder / "SKILL.md"
    path.write_text("---\nname: example\ndescription: example\n---\nOriginal", encoding="utf-8")
    manager = SkillManager(str(tmp_path))
    first = manager.load_skill("example")
    assert first["instructions"] == "Original"
    path.write_text("---\nname: example\ndescription: example\n---\nUpdated", encoding="utf-8")
    assert manager.load_skill("example")["error_code"] == "SKILL_CHANGED_OR_UNAVAILABLE"
    refreshed = SkillManager(str(tmp_path)).load_skill("example")
    assert refreshed["instructions"] == "Updated"
    assert refreshed["version"] != first["version"]
    path.unlink()
    assert manager.load_skill("example")["success"] is False


def test_invalid_and_oversized_files_do_not_break_discovery(tmp_path):
    for name, content in [("wrong", "---\nname: different\ndescription: test\n---\nbody"),
                          ("huge", "a" * 24001), ("empty", "---\nname: empty\ndescription: test\n---\n")]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "SKILL.md").write_text(content, encoding="utf-8")
    assert SkillManager(str(tmp_path)).skill_count == 0
    assert SkillManager(enabled=False).load_skill("anything")["error_code"] == "SKILLS_DISABLED"
