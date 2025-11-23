"""WebSocket server for real-time data streaming.

Provides WebSocket connections for live EEG streaming,
event notifications, and bi-directional communication.
"""

from datetime import datetime
from typing import Any
import json
import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect


logger = logging.getLogger(__name__)
router = APIRouter()


class WebSocketManager:
    """Manage WebSocket connections.

    Handles multiple client connections and broadcasts
    real-time data updates.

    Example:
        >>> manager = WebSocketManager()
        >>> await manager.connect(websocket)
        >>> await manager.broadcast({"type": "data", "value": [1,2,3]})
    """

    def __init__(self):
        """Initialize WebSocket manager."""
        self._connections: dict[str, WebSocket] = {}
        self._subscriptions: dict[str, set[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept a new connection.

        Args:
            websocket: WebSocket connection.
            client_id: Client identifier.
        """
        await websocket.accept()
        self._connections[client_id] = websocket
        self._subscriptions[client_id] = set()

        logger.info(f"WebSocket connected: {client_id}")

        await self.send(client_id, {
            "type": "connected",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
        })

    def disconnect(self, client_id: str) -> None:
        """Handle disconnection.

        Args:
            client_id: Client identifier.
        """
        if client_id in self._connections:
            del self._connections[client_id]
        if client_id in self._subscriptions:
            del self._subscriptions[client_id]

        logger.info(f"WebSocket disconnected: {client_id}")

    async def send(self, client_id: str, message: dict[str, Any]) -> bool:
        """Send message to specific client.

        Args:
            client_id: Client identifier.
            message: Message to send.

        Returns:
            True if successful.
        """
        if client_id not in self._connections:
            return False

        try:
            await self._connections[client_id].send_json(message)
            return True
        except Exception as e:
            logger.error(f"Error sending to {client_id}: {e}")
            return False

    async def broadcast(
        self,
        message: dict[str, Any],
        channel: str | None = None,
    ) -> int:
        """Broadcast message to all or subscribed clients.

        Args:
            message: Message to broadcast.
            channel: Optional channel filter.

        Returns:
            Number of clients sent to.
        """
        sent = 0

        for client_id, websocket in list(self._connections.items()):
            # Check subscription
            if channel:
                if client_id not in self._subscriptions:
                    continue
                if channel not in self._subscriptions[client_id]:
                    continue

            try:
                await websocket.send_json(message)
                sent += 1
            except Exception as e:
                logger.error(f"Broadcast error for {client_id}: {e}")
                self.disconnect(client_id)

        return sent

    def subscribe(self, client_id: str, channel: str) -> None:
        """Subscribe client to a channel.

        Args:
            client_id: Client identifier.
            channel: Channel name.
        """
        if client_id not in self._subscriptions:
            self._subscriptions[client_id] = set()
        self._subscriptions[client_id].add(channel)

    def unsubscribe(self, client_id: str, channel: str) -> None:
        """Unsubscribe client from a channel.

        Args:
            client_id: Client identifier.
            channel: Channel name.
        """
        if client_id in self._subscriptions:
            self._subscriptions[client_id].discard(channel)

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self._connections)

    def get_clients(self) -> list[str]:
        """Get list of connected client IDs."""
        return list(self._connections.keys())


# Global manager instance
_ws_manager: WebSocketManager | None = None


def get_ws_manager() -> WebSocketManager:
    """Get the global WebSocket manager.

    Returns:
        Global WebSocketManager instance.
    """
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager


# WebSocket endpoints
@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """Main WebSocket endpoint for data streaming."""
    import uuid
    client_id = str(uuid.uuid4())[:8]
    manager = get_ws_manager()

    try:
        await manager.connect(websocket, client_id)

        while True:
            # Receive message
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "subscribe":
                channel = data.get("channel", "")
                manager.subscribe(client_id, channel)
                await manager.send(client_id, {
                    "type": "subscribed",
                    "channel": channel,
                })

            elif msg_type == "unsubscribe":
                channel = data.get("channel", "")
                manager.unsubscribe(client_id, channel)
                await manager.send(client_id, {
                    "type": "unsubscribed",
                    "channel": channel,
                })

            elif msg_type == "ping":
                await manager.send(client_id, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat(),
                })

            else:
                # Echo unknown messages
                await manager.send(client_id, {
                    "type": "echo",
                    "data": data,
                })

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(client_id)


@router.websocket("/eeg/{session_id}")
async def websocket_eeg(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for live EEG streaming."""
    import uuid
    client_id = str(uuid.uuid4())[:8]
    manager = get_ws_manager()

    try:
        await manager.connect(websocket, client_id)
        manager.subscribe(client_id, f"eeg:{session_id}")

        await manager.send(client_id, {
            "type": "eeg_stream_started",
            "session_id": session_id,
        })

        while True:
            data = await websocket.receive_json()

            if data.get("type") == "stop":
                break

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(client_id)


@router.websocket("/events")
async def websocket_events(websocket: WebSocket):
    """WebSocket endpoint for event notifications."""
    import uuid
    client_id = str(uuid.uuid4())[:8]
    manager = get_ws_manager()

    try:
        await manager.connect(websocket, client_id)
        manager.subscribe(client_id, "events")

        while True:
            await asyncio.sleep(1)  # Keep connection alive

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(client_id)
