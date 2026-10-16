"""Reference-counted MCP connection shared by agent sessions in one process."""

from __future__ import annotations

import threading

from app.mcp_client.client import MCPClient


_lock = threading.Lock()
_entries: dict[str, tuple[MCPClient, list[dict], int]] = {}
_retired: dict[int, tuple[MCPClient, int]] = {}


def acquire_client(url: str) -> tuple[MCPClient, list[dict]]:
    with _lock:
        if url in _entries:
            client, definitions, users = _entries[url]
            _entries[url] = (client, definitions, users + 1)
            return client, definitions
        client = MCPClient(url)
        try:
            definitions = client.connect()
        except BaseException:
            client.close()
            raise
        _entries[url] = (client, definitions, 1)
        return client, definitions


def release_client(url: str, client: MCPClient) -> None:
    close = False
    with _lock:
        entry = _entries.get(url)
        if entry is not None and entry[0] is client:
            _, definitions, users = entry
            if users > 1:
                _entries[url] = (client, definitions, users - 1)
            else:
                del _entries[url]
                close = True
        else:
            retired = _retired.get(id(client))
            if retired is None or retired[0] is not client:
                return
            if retired[1] > 1:
                _retired[id(client)] = (client, retired[1] - 1)
            else:
                del _retired[id(client)]
                close = True
    if close:
        client.close()


def invalidate_client(url: str, client: MCPClient) -> None:
    close = False
    with _lock:
        entry = _entries.get(url)
        if entry and entry[0] is client:
            del _entries[url]
            remaining = entry[2] - 1
            if remaining:
                _retired[id(client)] = (client, remaining)
            else:
                close = True
        else:
            retired = _retired.get(id(client))
            if retired and retired[0] is client:
                remaining = retired[1] - 1
                if remaining:
                    _retired[id(client)] = (client, remaining)
                else:
                    del _retired[id(client)]
                    close = True
    if close:
        client.close()
