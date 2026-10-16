"""Post-turn summary outages must preserve the completed business response."""
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import APITimeoutError, APIConnectionError, RateLimitError

from app.agent.memory.session_service import MemorySessionService
from app.schemas.response import CustomerServiceResponse, IntentType


@pytest.mark.parametrize("kind", ["timeout", "connection", "rate_limit", "bug"])
def test_auto_summary_outage_preserves_completed_turn(kind):
    request = httpx.Request("POST", "https://model.invalid")
    error = {
        "timeout": APITimeoutError(request=request),
        "connection": APIConnectionError(request=request),
        "rate_limit": RateLimitError("limited", response=httpx.Response(429, request=request), body=None),
        "bug": ValueError("invalid summary"),
    }[kind]
    result = CustomerServiceResponse(intent=IntentType.ORDER_QUERY, confidence=1, reply="订单记录已发货")
    memory = Mock()
    memory.stm_to_dict.return_value = {}
    owner = SimpleNamespace(raw_messages=[], summary="old", summary_versions=[], summary_audit=[],
                            memory_manager=memory, history_threshold=0, session_path="unused",
                            _execute_mode_turn=Mock(return_value=result))
    service = MemorySessionService(owner)
    service.compress_history = Mock(side_effect=error)
    service.save = Mock()
    if kind == "bug":
        with pytest.raises(ValueError):
            service.execute_turn("查订单")
        assert owner.raw_messages == []
        service.save.assert_not_called()
    else:
        assert service.execute_turn("查订单") is result
        assert len(owner.raw_messages) == 2
        assert owner.summary == "old"
        assert "订单记录已发货" in owner.raw_messages[-1]["content"]
        service.save.assert_called_once()
        memory.commit_long_term_checkpoint.assert_called_once()
        memory.discard_staged_checkpoint.assert_not_called()
