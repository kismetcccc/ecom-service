"""Trusted return-skill adapter. Text/resources cannot authorize a transaction."""
from uuid import uuid4

from app.agent.skills.loader import SkillManager
from app.config.settings import settings

RUNTIME_VERSION = "return-v2"


def load_return_skill(manager=None, available_tools=None, resource=""):
    manager = manager if manager is not None else SkillManager(settings.skills_dir, settings.skills_enabled)
    return manager.load_skill("process-return", available_tools=available_tools, resource=resource)


def execution_snapshot(loaded, state, *, run_id=None, policy_scope="system_precheck_only"):
    return {"name": "process-return", "run_id": run_id or uuid4().hex,
            "version": loaded.get("version", ""), "bundle_version": loaded.get("bundle_version", ""),
            "runtime_version": RUNTIME_VERSION, "state": state, "policy_scope": policy_scope,
            "resources": [], "reason_code": ""}
