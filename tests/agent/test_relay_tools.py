"""Tests for the core Relay-managed Hermes tool adapter."""

from __future__ import annotations

import asyncio
import contextvars
import json

import pytest

pytest.importorskip("nemo_relay")

from agent import relay_runtime, relay_tools


@pytest.fixture()
def relay_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="session-1",
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="turn-1",
        task_id="task-1",
    )
    lease.host.retain_managed_execution("test.relay_tools")
    try:
        yield lease.host.relay
    finally:
        lease.host.release_managed_execution("test.relay_tools")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


def test_tool_adapter_bypasses_relay_without_an_active_consumer(
    relay_turn, monkeypatch
):
    relay = relay_turn
    runtime = relay_runtime.get_runtime()
    assert runtime is not None
    runtime.release_managed_execution("test.relay_tools")
    runtime.release_managed_execution(
        relay_runtime.RELAY_PLUGINS_EXECUTION_CONSUMER
    )
    args = {"command": "pwd"}

    monkeypatch.setattr(
        relay.tools,
        "execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("inactive Relay must not manage the tool call")
        ),
    )

    result, final_args = relay_tools.execute(
        "terminal",
        args,
        lambda value: value,
        session_id="session-1",
    )

    assert result is args
    assert final_args is args




def test_request_rewrite_reaches_authorized_callback_once(relay_turn):
    relay = relay_turn
    callback_args = []

    def rewrite_request(_name, args):
        return {**args, "path": "/approved/path"}

    async def wrap_execution(_name, args, next_call):
        result = await next_call(args)
        outcome = {**getattr(result, "result", result), "wrapped": True}
        if hasattr(result, "annotation"):
            return relay.ToolExecutionInterceptOutcome(
                outcome, annotation=result.annotation
            )
        return relay.ToolExecutionInterceptOutcome(outcome)

    relay.intercepts.register_tool_request(
        "hermes-test-tool-request", 1, False, rewrite_request
    )
    relay.intercepts.register_tool_execution(
        "hermes-test-tool-execution", 1, wrap_execution
    )
    try:
        result, observed_args = relay_tools.execute(
            "write_file",
            {"path": "/original/path"},
            lambda args: callback_args.append(args) or {"ok": True},
            session_id="session-1",
            metadata={"tool_call_id": "call-1"},
        )
    finally:
        relay.intercepts.deregister_tool_execution("hermes-test-tool-execution")
        relay.intercepts.deregister_tool_request("hermes-test-tool-request")

    assert callback_args == [{"path": "/approved/path"}]
    assert observed_args == {"path": "/approved/path"}
    assert isinstance(result, str)
    assert json.loads(result) == {"ok": True, "wrapped": True}






def test_tool_error_is_preserved_from_relay_wrapper_suffix(relay_turn, monkeypatch):
    relay = relay_turn

    class ToolError(Exception):
        pass

    tool_error = ToolError("dispatch failed")

    async def wrapping_execute(_name, args, callback, **_kwargs):
        try:
            return callback(args)
        except Exception as exc:
            raise RuntimeError(
                f"internal error: {type(exc).__name__}: {exc} (worker trace)"
            ) from None

    monkeypatch.setattr(relay.tools, "execute", wrapping_execute)

    with pytest.raises(ToolError) as caught:
        relay_tools.execute(
            "terminal",
            {"command": "false"},
            lambda _args: (_ for _ in ()).throw(tool_error),
            session_id="session-1",
        )

    assert caught.value is tool_error


def test_tool_adapter_wraps_callback_and_unwraps_cached_result(relay_turn, monkeypatch):
    relay = relay_turn
    if not hasattr(relay, "ToolExecutionResult"):
        pytest.skip("canonical tool results require NeMo Relay 0.8")
    seen = []

    async def managed_execute(_name, args, callback, **_kwargs):
        wrapped = callback(args)
        seen.append(wrapped)
        return relay.ToolExecutionResult({"cached": True})

    monkeypatch.setattr(relay.tools, "execute", managed_execute)

    result, final_args = relay_tools.execute(
        "read_only_lookup",
        {"query": "cache"},
        lambda _args: {"live": True},
        session_id="session-1",
    )

    assert isinstance(seen[0], relay.ToolExecutionResult)
    assert seen[0].result == {"live": True}
    assert result == '{"cached": true}'
    assert final_args == {"query": "cache"}


def test_read_only_tool_cache_hit_skips_hermes_callback(relay_turn):
    relay = relay_turn
    if not hasattr(relay, "ToolExecutionResult"):
        pytest.skip("tool-result caching requires NeMo Relay 0.8")

    async def activate_cache():
        await relay.plugin.initialize(
            {
                "components": [
                    {
                        "kind": "adaptive",
                        "enabled": True,
                        "config": {
                            "version": 1,
                            "response_cache": {
                                "namespace": "hermes-tool-cache-test",
                                "tools": {
                                    "enabled": True,
                                    "classes": {
                                        "read_only": {
                                            "cacheable": True,
                                            "members": ["read_only_lookup"],
                                        }
                                    },
                                },
                            },
                        },
                    }
                ]
            }
        )

    asyncio.run(activate_cache())
    calls = 0

    def tool(_args):
        nonlocal calls
        calls += 1
        return "live result"

    try:
        first, _ = relay_tools.execute(
            "read_only_lookup",
            {"query": "cache"},
            tool,
            session_id="session-1",
        )
        second, _ = relay_tools.execute(
            "read_only_lookup",
            {"query": "cache"},
            tool,
            session_id="session-1",
        )
    finally:
        asyncio.run(relay.plugin.clear_async())

    assert calls == 1
    assert first == second == "live result"
