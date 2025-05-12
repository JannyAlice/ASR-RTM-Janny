from typing import Any, Callable, Dict, List, Set
from dataclasses import dataclass, field
import logging
import traceback

logger = logging.getLogger(__name__)

class PluginEventError(Exception):
    """插件事件系统异常基类"""
    pass

class InvalidEventTypeError(PluginEventError):
    """无效的事件类型错误"""
    pass

class EventHandlerError(PluginEventError):
    """事件处理器错误"""
    pass

@dataclass
class PluginEvent:
    """插件事件类，用于在插件间传递事件"""
    event_type: str
    source: str
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """数据校验"""
        if not self.event_type:
            raise InvalidEventTypeError("Event type cannot be empty")
        if not self.source:
            raise InvalidEventTypeError("Event source cannot be empty")

class PluginEventSystem:
    """插件事件系统，处理插件间的事件传递"""
    
    def __init__(self):
        self._subscribers: Dict[str, Set[Callable[[PluginEvent], None]]] = {}
        logger.info("Plugin event system initialized")
        
    def subscribe(self, event_type: str, callback: Callable[[PluginEvent], None]) -> None:
        """订阅事件
        
        Args:
            event_type: 事件类型
            callback: 回调函数
            
        Raises:
            InvalidEventTypeError: 事件类型无效
        """
        try:
            if not event_type:
                raise InvalidEventTypeError("Event type cannot be empty")
            if not callback:
                raise InvalidEventTypeError("Callback function cannot be None")
                
            if event_type not in self._subscribers:
                self._subscribers[event_type] = set()
            self._subscribers[event_type].add(callback)
            logger.debug(f"Subscribed to event type: {event_type}")
            
        except PluginEventError as e:
            logger.error(f"Error subscribing to event: {str(e)}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error subscribing to event: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise PluginEventError(error_msg)
            
    def unsubscribe(self, event_type: str, callback: Callable[[PluginEvent], None]) -> None:
        """取消订阅事件
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        try:
            if event_type in self._subscribers:
                self._subscribers[event_type].discard(callback)
                logger.debug(f"Unsubscribed from event type: {event_type}")
                
                # 如果没有订阅者了，删除该事件类型
                if not self._subscribers[event_type]:
                    del self._subscribers[event_type]
                    logger.debug(f"Removed empty event type: {event_type}")
                    
        except Exception as e:
            logger.error(f"Error unsubscribing from event: {str(e)}")
            logger.error(traceback.format_exc())
            
    def emit(self, event: PluginEvent) -> None:
        """发送事件
        
        Args:
            event: 要发送的事件
            
        Raises:
            InvalidEventTypeError: 事件无效
            EventHandlerError: 事件处理器错误
        """
        try:
            if not isinstance(event, PluginEvent):
                raise InvalidEventTypeError("Invalid event object")
                
            if event.event_type in self._subscribers:
                logger.debug(f"Emitting event: {event.event_type} from {event.source}")
                for callback in self._subscribers[event.event_type]:
                    try:
                        callback(event)
                    except Exception as e:
                        error_msg = f"Error in event handler for {event.event_type}: {str(e)}"
                        logger.error(error_msg)
                        logger.error(traceback.format_exc())
                        raise EventHandlerError(error_msg)
            else:
                logger.debug(f"No subscribers for event type: {event.event_type}")
                
        except PluginEventError:
            raise
        except Exception as e:
            error_msg = f"Unexpected error emitting event: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise PluginEventError(error_msg)
            
    def clear(self) -> None:
        """清除所有订阅"""
        try:
            event_types = list(self._subscribers.keys())
            self._subscribers.clear()
            logger.info(f"Cleared all subscriptions for event types: {event_types}")
        except Exception as e:
            logger.error(f"Error clearing subscriptions: {str(e)}")
            logger.error(traceback.format_exc())