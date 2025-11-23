"""Python client library for the workbench API.

Provides easy access to the REST API and WebSocket
streaming capabilities.
"""

from datetime import datetime
from typing import Any, AsyncGenerator
import json
import logging


logger = logging.getLogger(__name__)


class WorkbenchClient:
    """Client for the Consciousness Research Workbench API.

    Provides methods for session management, analysis,
    and real-time data streaming.

    Example:
        >>> client = WorkbenchClient("http://localhost:8000")
        >>> session = client.create_session("my_session")
        >>> client.start_session(session["session_id"])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize client.

        Args:
            base_url: API base URL.
            api_key: Optional API key.
            timeout: Request timeout.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = None

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request.

        Args:
            method: HTTP method.
            endpoint: API endpoint.
            data: Request body.
            params: Query parameters.

        Returns:
            Response data.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests required: pip install requests")

        url = f"{self.base_url}/api/v1{endpoint}"

        response = requests.request(
            method=method,
            url=url,
            headers=self._get_headers(),
            json=data,
            params=params,
            timeout=self.timeout,
        )

        response.raise_for_status()
        return response.json()

    # Session methods
    def create_session(
        self,
        name: str,
        device: str = "muse_2",
        scenario: str = "free",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new session.

        Args:
            name: Session name.
            device: EEG device type.
            scenario: Scenario name.
            metadata: Additional metadata.

        Returns:
            Session data.
        """
        return self._request("POST", "/sessions", data={
            "name": name,
            "device": device,
            "scenario": scenario,
            "metadata": metadata or {},
        })

    def list_sessions(self, limit: int = 10, offset: int = 0) -> list[dict]:
        """List sessions.

        Args:
            limit: Maximum results.
            offset: Result offset.

        Returns:
            List of sessions.
        """
        return self._request("GET", "/sessions", params={
            "limit": limit,
            "offset": offset,
        })

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Get session details.

        Args:
            session_id: Session ID.

        Returns:
            Session data.
        """
        return self._request("GET", f"/sessions/{session_id}")

    def start_session(self, session_id: str) -> dict[str, Any]:
        """Start recording.

        Args:
            session_id: Session ID.

        Returns:
            Status response.
        """
        return self._request("POST", f"/sessions/{session_id}/start")

    def stop_session(self, session_id: str) -> dict[str, Any]:
        """Stop recording.

        Args:
            session_id: Session ID.

        Returns:
            Status response.
        """
        return self._request("POST", f"/sessions/{session_id}/stop")

    def add_marker(
        self,
        session_id: str,
        label: str,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add event marker.

        Args:
            session_id: Session ID.
            label: Marker label.
            timestamp: Marker timestamp.
            metadata: Additional metadata.

        Returns:
            Status response.
        """
        return self._request("POST", f"/sessions/{session_id}/markers", data={
            "label": label,
            "timestamp": timestamp,
            "metadata": metadata or {},
        })

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete session.

        Args:
            session_id: Session ID.

        Returns:
            Status response.
        """
        return self._request("DELETE", f"/sessions/{session_id}")

    # Analysis methods
    def run_analysis(
        self,
        session_id: str,
        analysis_type: str = "full",
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run analysis on session.

        Args:
            session_id: Session ID.
            analysis_type: Type of analysis.
            parameters: Analysis parameters.

        Returns:
            Analysis results.
        """
        return self._request("POST", "/analysis", data={
            "session_id": session_id,
            "analysis_type": analysis_type,
            "parameters": parameters or {},
        })

    def get_analysis(self, analysis_id: str) -> dict[str, Any]:
        """Get analysis results.

        Args:
            analysis_id: Analysis ID.

        Returns:
            Analysis data.
        """
        return self._request("GET", f"/analysis/{analysis_id}")

    # Consciousness methods
    def classify_state(
        self,
        features: list[float],
        include_confidence: bool = True,
    ) -> dict[str, Any]:
        """Classify consciousness state.

        Args:
            features: Feature vector.
            include_confidence: Include confidence scores.

        Returns:
            Classification result.
        """
        return self._request("POST", "/consciousness/classify", data={
            "features": features,
            "include_confidence": include_confidence,
        })

    def list_states(self) -> dict[str, Any]:
        """List available consciousness states.

        Returns:
            States dictionary.
        """
        return self._request("GET", "/consciousness/states")

    # Report methods
    def generate_report(
        self,
        session_id: str,
        format: str = "html",
    ) -> dict[str, Any]:
        """Generate session report.

        Args:
            session_id: Session ID.
            format: Report format.

        Returns:
            Report info.
        """
        return self._request("POST", f"/reports/session/{session_id}", params={
            "format": format,
        })

    # System methods
    def health(self) -> dict[str, Any]:
        """Check API health.

        Returns:
            Health status.
        """
        return self._request("GET", "/health")

    def system_info(self) -> dict[str, Any]:
        """Get system info.

        Returns:
            System information.
        """
        return self._request("GET", "/system/info")

    def list_devices(self) -> dict[str, Any]:
        """List supported devices.

        Returns:
            Devices list.
        """
        return self._request("GET", "/devices")


class AsyncWorkbenchClient:
    """Async client for the API.

    Provides async methods and WebSocket streaming.

    Example:
        >>> async with AsyncWorkbenchClient() as client:
        ...     sessions = await client.list_sessions()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
    ):
        """Initialize async client.

        Args:
            base_url: API base URL.
            api_key: Optional API key.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session = None

    async def __aenter__(self) -> "AsyncWorkbenchClient":
        """Async context manager entry."""
        try:
            import aiohttp
            self._session = aiohttp.ClientSession()
        except ImportError:
            raise ImportError("aiohttp required: pip install aiohttp")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
    ) -> dict[str, Any]:
        """Make async HTTP request."""
        if not self._session:
            raise RuntimeError("Client not initialized. Use 'async with'.")

        url = f"{self.base_url}/api/v1{endpoint}"

        async with self._session.request(
            method=method,
            url=url,
            headers=self._get_headers(),
            json=data,
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def list_sessions(self) -> list[dict]:
        """List sessions asynchronously."""
        return await self._request("GET", "/sessions")

    async def stream_eeg(
        self,
        session_id: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream EEG data via WebSocket.

        Args:
            session_id: Session ID.

        Yields:
            Data packets.
        """
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets required: pip install websockets")

        ws_url = self.base_url.replace("http", "ws")
        uri = f"{ws_url}/ws/eeg/{session_id}"

        async with websockets.connect(uri) as websocket:
            while True:
                try:
                    message = await websocket.recv()
                    yield json.loads(message)
                except websockets.ConnectionClosed:
                    break
