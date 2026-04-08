"""Base Agent — 所有角色的基礎類"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

from models.messages import AgentMessage, MessageType


class BaseAgent(ABC):
    """
    Agent 基礎類，提供：
    - 消息收發（透過 asyncio.Queue）
    - 生命週期管理 (start / stop)
    - 日誌
    - 狀態管理
    """

    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.inbox: Optional[asyncio.Queue] = None
        self._running = False
        self._bus: Optional[MessageBus] = None
        self.state: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"agent.{agent_id}")

    # ── 生命週期 ─────────────────────────────

    async def start(self):
        """啟動 agent，進入消息處理循環"""
        self.inbox = asyncio.Queue()
        self._running = True
        self.logger.info(f"[{self.name}] 啟動")
        await self.on_start()
        asyncio.create_task(self._message_loop())

    async def stop(self):
        """停止 agent"""
        self._running = False
        self.logger.info(f"[{self.name}] 停止")
        await self.on_stop()

    async def on_start(self):
        """子類可覆寫的啟動鉤子"""
        pass

    async def on_stop(self):
        """子類可覆寫的停止鉤子"""
        pass

    # ── 消息處理 ─────────────────────────────

    async def _message_loop(self):
        """持續從 inbox 取出消息並處理"""
        while self._running:
            try:
                msg = await asyncio.wait_for(self.inbox.get(), timeout=1.0)
                self.logger.debug(f"[{self.name}] 收到消息: {msg.msg_type} from {msg.sender}")
                await self.handle_message(msg)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"[{self.name}] 處理消息錯誤: {e}", exc_info=True)

    @abstractmethod
    async def handle_message(self, message: AgentMessage):
        """處理收到的消息 — 子類必須實作"""
        ...

    async def send(self, receiver: str, msg_type: MessageType, payload: dict, priority: int = 0):
        """發送消息給其他 agent"""
        msg = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            msg_type=msg_type,
            payload=payload,
            priority=priority,
        )
        if self._bus:
            await self._bus.publish(msg)
        else:
            self.logger.warning(f"[{self.name}] 未連接 MessageBus，消息未發送")

    async def broadcast(self, msg_type: MessageType, payload: dict, priority: int = 0):
        """廣播消息給所有 agent"""
        await self.send("broadcast", msg_type, payload, priority)

    # ── 狀態 ─────────────────────────────────

    def get_status(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "running": self._running,
            "inbox_size": self.inbox.qsize() if self.inbox else 0,
            "state": self.state,
        }


class MessageBus:
    """
    簡易消息匯流排，負責在 agent 之間路由消息。
    """

    def __init__(self):
        self.agents: dict[str, BaseAgent] = {}
        self.message_log: list[AgentMessage] = []
        self.logger = logging.getLogger("message_bus")

    def register(self, agent: BaseAgent):
        """註冊 agent 到匯流排"""
        self.agents[agent.agent_id] = agent
        agent._bus = self
        self.logger.info(f"已註冊: {agent.name} ({agent.agent_id})")

    async def publish(self, message: AgentMessage):
        """發佈消息到目標 agent 或廣播"""
        self.message_log.append(message)
        # 防止無限增長
        if len(self.message_log) > 10000:
            self.message_log = self.message_log[-5000:]

        if message.receiver == "broadcast":
            for aid, agent in self.agents.items():
                if aid != message.sender:
                    await agent.inbox.put(message)
        elif message.receiver in self.agents:
            await self.agents[message.receiver].inbox.put(message)
        else:
            self.logger.warning(f"目標 agent 不存在: {message.receiver}")

    async def start_all(self):
        """啟動所有已註冊的 agent"""
        for agent in self.agents.values():
            await agent.start()

    async def stop_all(self):
        """停止所有 agent"""
        for agent in self.agents.values():
            await agent.stop()
