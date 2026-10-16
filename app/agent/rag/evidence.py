"""Attach only tool-verified, current-turn policy evidence to chat responses."""

import json

from app.schemas.response import CustomerServiceResponse, PolicySource


def attach_knowledge_evidence(
    result: CustomerServiceResponse, messages: list[dict], *,
    preserve_reply_on_failure: bool = False,
) -> CustomerServiceResponse:
    """Never accept model-generated source IDs or citations as verified evidence."""
    result.sources = []
    result.knowledge_status = None
    result.knowledge_providers = []
    result.knowledge_fallback = False
    call_ids = {
        call.get("id")
        for message in messages if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
        if call.get("function", {}).get("name") == "search_knowledge"
    }
    seen: set[str] = set()
    failure = ""
    for message in messages:
        if message.get("role") != "tool" or message.get("tool_call_id") not in call_ids:
            continue
        try:
            data = json.loads(message.get("content", ""))
        except (TypeError, ValueError):
            failure = "BAD_RESPONSE"
            continue
        if not isinstance(data, dict) or type(data.get("success")) is not bool:
            failure = "BAD_RESPONSE"
            continue
        provider = data.get("provider")
        if provider in {"rag", "wiki"} and provider not in result.knowledge_providers:
            result.knowledge_providers.append(provider)
        result.knowledge_fallback |= bool(data.get("wiki_fallback_reason"))
        if data["success"] is False:
            failure = data.get("error_code", "UNAVAILABLE")
            continue
        if not isinstance(data.get("results"), list):
            failure = "BAD_RESPONSE"
            continue
        if not data["results"]:
            failure = "NO_MATCH"
            continue
        for hit in data.get("results", []):
            if not isinstance(hit, dict):
                continue
            chunk_id = str(hit.get("chunk_id", ""))
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            result.sources.append(PolicySource(
                chunk_id=chunk_id, doc=str(hit.get("doc", "")),
                section=str(hit.get("section", "")),
                version=str(hit.get("version", "")),
                effective_from=str(hit.get("effective_from", "")),
                source_url=str(hit.get("source_url", "")),
                is_demo=hit.get("status") == "demo",
            ))
    if failure and not result.sources:
        # A failed lookup cannot substantiate an LLM's specific policy claim.
        result.knowledge_status = failure
        if not preserve_reply_on_failure:
            result.reply = "当前未能核实有效政策依据。请稍后重试，或联系人工客服确认具体规则。"
    elif result.sources:
        result.knowledge_status = "VERIFIED_DEMO" if any(s.is_demo for s in result.sources) else "VERIFIED"
    return result
