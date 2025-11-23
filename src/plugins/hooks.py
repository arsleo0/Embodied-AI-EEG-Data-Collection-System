"""Event hook system for plugin integration.

Provides a flexible hook system for plugins to integrate
with the data processing pipeline and UI.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import logging
import asyncio
from functools import wraps


logger = logging.getLogger(__name__)


class HookType(Enum):
    """Types of hooks available in the system."""

    # Data pipeline hooks
    PRE_PROCESS = "pre_process"
    POST_PROCESS = "post_process"
    PRE_ANALYSIS = "pre_analysis"
    POST_ANALYSIS = "post_analysis"

    # Session hooks
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    MARKER_ADDED = "marker_added"

    # Data hooks
    DATA_RECEIVED = "data_received"
    DATA_SAVED = "data_saved"
    DATA_LOADED = "data_loaded"

    # Visualization hooks
    PRE_RENDER = "pre_render"
    POST_RENDER = "post_render"
    FIGURE_CREATED = "figure_created"

    # Classification hooks
    PRE_CLASSIFY = "pre_classify"
    POST_CLASSIFY = "post_classify"

    # Report hooks
    PRE_REPORT = "pre_report"
    POST_REPORT = "post_report"

    # System hooks
    PLUGIN_LOADED = "plugin_loaded"
    PLUGIN_UNLOADED = "plugin_unloaded"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class HookCallback:
    """A registered hook callback.

    Attributes:
        name: Callback name.
        callback: The callable.
        priority: Execution priority (lower = earlier).
        plugin_name: Name of plugin that registered this.
        is_async: Whether callback is async.
    """

    name: str
    callback: Callable
    priority: int = 100
    plugin_name: str = ""
    is_async: bool = False


@dataclass
class HookResult:
    """Result from hook execution.

    Attributes:
        hook_type: Type of hook executed.
        success: Whether all callbacks succeeded.
        results: Results from each callback.
        errors: Any errors that occurred.
        modified_data: Data modified by hooks.
    """

    hook_type: HookType
    success: bool
    results: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    modified_data: Any = None


class HookManager:
    """Manages hook registration and execution.

    Central system for registering and invoking hooks
    throughout the application lifecycle.

    Example:
        >>> manager = HookManager()
        >>> manager.register(HookType.PRE_PROCESS, my_callback)
        >>> result = manager.invoke(HookType.PRE_PROCESS, data=eeg_data)
    """

    def __init__(self):
        """Initialize hook manager."""
        self._hooks: dict[HookType, list[HookCallback]] = {
            hook_type: [] for hook_type in HookType
        }
        self._enabled = True

    def register(
        self,
        hook_type: HookType,
        callback: Callable,
        name: str | None = None,
        priority: int = 100,
        plugin_name: str = "",
    ) -> None:
        """Register a hook callback.

        Args:
            hook_type: Type of hook to register for.
            callback: Callback function.
            name: Optional callback name.
            priority: Execution priority.
            plugin_name: Name of registering plugin.
        """
        if name is None:
            name = callback.__name__

        hook_callback = HookCallback(
            name=name,
            callback=callback,
            priority=priority,
            plugin_name=plugin_name,
            is_async=asyncio.iscoroutinefunction(callback),
        )

        self._hooks[hook_type].append(hook_callback)
        self._hooks[hook_type].sort(key=lambda x: x.priority)

        logger.debug(f"Registered hook: {hook_type.value}/{name}")

    def unregister(
        self,
        hook_type: HookType,
        name: str | None = None,
        plugin_name: str | None = None,
    ) -> int:
        """Unregister hook callbacks.

        Args:
            hook_type: Type of hook.
            name: Callback name to remove.
            plugin_name: Remove all from this plugin.

        Returns:
            Number of callbacks removed.
        """
        original_count = len(self._hooks[hook_type])

        if name:
            self._hooks[hook_type] = [
                h for h in self._hooks[hook_type] if h.name != name
            ]
        elif plugin_name:
            self._hooks[hook_type] = [
                h for h in self._hooks[hook_type]
                if h.plugin_name != plugin_name
            ]

        removed = original_count - len(self._hooks[hook_type])
        if removed:
            logger.debug(f"Unregistered {removed} hooks from {hook_type.value}")

        return removed

    def invoke(
        self,
        hook_type: HookType,
        data: Any = None,
        **kwargs,
    ) -> HookResult:
        """Invoke all callbacks for a hook.

        Args:
            hook_type: Type of hook to invoke.
            data: Data to pass to callbacks.
            **kwargs: Additional arguments.

        Returns:
            HookResult with execution details.
        """
        if not self._enabled:
            return HookResult(
                hook_type=hook_type,
                success=True,
                modified_data=data,
            )

        results = []
        errors = []
        modified_data = data

        for hook in self._hooks[hook_type]:
            try:
                if hook.is_async:
                    # Run async in new event loop if needed
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    result = loop.run_until_complete(
                        hook.callback(modified_data, **kwargs)
                    )
                else:
                    result = hook.callback(modified_data, **kwargs)

                results.append(result)

                # Allow hooks to modify data
                if result is not None and isinstance(result, type(data)):
                    modified_data = result

            except Exception as e:
                error_msg = f"{hook.name}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Hook error in {hook_type.value}/{hook.name}: {e}")

        return HookResult(
            hook_type=hook_type,
            success=len(errors) == 0,
            results=results,
            errors=errors,
            modified_data=modified_data,
        )

    async def invoke_async(
        self,
        hook_type: HookType,
        data: Any = None,
        **kwargs,
    ) -> HookResult:
        """Asynchronously invoke all callbacks for a hook.

        Args:
            hook_type: Type of hook to invoke.
            data: Data to pass to callbacks.
            **kwargs: Additional arguments.

        Returns:
            HookResult with execution details.
        """
        if not self._enabled:
            return HookResult(
                hook_type=hook_type,
                success=True,
                modified_data=data,
            )

        results = []
        errors = []
        modified_data = data

        for hook in self._hooks[hook_type]:
            try:
                if hook.is_async:
                    result = await hook.callback(modified_data, **kwargs)
                else:
                    result = hook.callback(modified_data, **kwargs)

                results.append(result)

                if result is not None and isinstance(result, type(data)):
                    modified_data = result

            except Exception as e:
                error_msg = f"{hook.name}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Hook error: {error_msg}")

        return HookResult(
            hook_type=hook_type,
            success=len(errors) == 0,
            results=results,
            errors=errors,
            modified_data=modified_data,
        )

    def get_hooks(self, hook_type: HookType) -> list[HookCallback]:
        """Get all registered hooks for a type.

        Args:
            hook_type: Type of hook.

        Returns:
            List of registered callbacks.
        """
        return self._hooks[hook_type].copy()

    def clear(self, hook_type: HookType | None = None) -> None:
        """Clear registered hooks.

        Args:
            hook_type: Specific hook type to clear, or all if None.
        """
        if hook_type:
            self._hooks[hook_type] = []
        else:
            for ht in HookType:
                self._hooks[ht] = []

    def enable(self) -> None:
        """Enable hook execution."""
        self._enabled = True

    def disable(self) -> None:
        """Disable hook execution."""
        self._enabled = False

    def list_all(self) -> dict[str, list[str]]:
        """List all registered hooks.

        Returns:
            Dictionary of hook types to callback names.
        """
        return {
            hook_type.value: [h.name for h in hooks]
            for hook_type, hooks in self._hooks.items()
            if hooks
        }


# Global hook manager
_global_manager: HookManager | None = None


def get_hook_manager() -> HookManager:
    """Get the global hook manager.

    Returns:
        Global HookManager instance.
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = HookManager()
    return _global_manager


def hook(
    hook_type: HookType,
    priority: int = 100,
    name: str | None = None,
) -> Callable:
    """Decorator to register a function as a hook.

    Args:
        hook_type: Type of hook.
        priority: Execution priority.
        name: Optional hook name.

    Returns:
        Decorator function.

    Example:
        >>> @hook(HookType.PRE_PROCESS, priority=50)
        ... def my_preprocessor(data):
        ...     return processed_data
    """
    def decorator(func: Callable) -> Callable:
        get_hook_manager().register(
            hook_type=hook_type,
            callback=func,
            name=name or func.__name__,
            priority=priority,
        )
        return func
    return decorator


def invoke_hook(hook_type: HookType, data: Any = None, **kwargs) -> HookResult:
    """Invoke a hook using the global manager.

    Args:
        hook_type: Type of hook.
        data: Data to pass.
        **kwargs: Additional arguments.

    Returns:
        HookResult.
    """
    return get_hook_manager().invoke(hook_type, data, **kwargs)
