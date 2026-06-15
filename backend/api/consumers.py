import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class ReconnectingConsumerMixin:
    """
    WebSocket 断线重连混合类
    - 服务端心跳（ping）
    - 客户端重连跟踪（通过 client_id）
    - 会话缓存（断线期间的告警暂存）
    """

    PING_INTERVAL = 30  # 秒
    PONG_TIMEOUT = 10  # 秒
    MAX_PENDING_ALERTS = 500  # 断线期间最大暂存告警数
    PENDING_TTL_SECONDS = 3600  # 暂存告警过期时间

    # 类级别缓存：{ client_id: { alerts: [...], last_seen: timestamp } }
    _pending_sessions = {}

    async def _cleanup_expired_sessions(self):
        """定期清理过期会话"""
        now = time.time()
        expired = [
            cid for cid, sess in self._pending_sessions.items()
            if now - sess['last_seen'] > self.PENDING_TTL_SECONDS
        ]
        for cid in expired:
            del self._pending_sessions[cid]

    def _track_session(self, client_id: str):
        """跟踪客户端会话"""
        if client_id not in self._pending_sessions:
            self._pending_sessions[client_id] = {
                'alerts': [],
                'last_seen': time.time(),
                'reconnect_count': 0
            }
        self._pending_sessions[client_id]['last_seen'] = time.time()
        return self._pending_sessions[client_id]

    def _queue_pending_alert(self, client_id: str, alert_data: dict):
        """断线期间暂存告警"""
        if client_id in self._pending_sessions:
            sess = self._pending_sessions[client_id]
            sess['alerts'].append({
                'data': alert_data,
                'queued_at': time.time()
            })
            # 限制队列大小
            if len(sess['alerts']) > self.MAX_PENDING_ALERTS:
                sess['alerts'] = sess['alerts'][-self.MAX_PENDING_ALERTS:]

    def _drain_pending_alerts(self, client_id: str):
        """客户端重连后取出所有暂存告警"""
        if client_id in self._pending_sessions:
            sess = self._pending_sessions[client_id]
            sess['reconnect_count'] += 1
            alerts = sess['alerts'][:]
            sess['alerts'] = []
            return alerts
        return []


class AlertConsumer(AsyncWebsocketConsumer, ReconnectingConsumerMixin):
    async def connect(self):
        # 从 query 参数获取 client_id（前端生成的唯一ID，用于重连跟踪）
        query_string = self.scope.get('query_string', b'').decode()
        self.client_id = None
        for kv in query_string.split('&'):
            if kv.startswith('client_id='):
                self.client_id = kv.split('=', 1)[1]
                break

        # 如果没有 client_id，生成一个临时的
        if not self.client_id:
            import uuid
            self.client_id = 'ws_' + uuid.uuid4().hex[:12]

        self.room_group_name = 'alerts'
        self.pong_received = True
        self.ping_task = None
        self.is_connected = False

        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

        # 跟踪会话
        sess = self._track_session(self.client_id)

        # 发送连接确认（带 client_id 给前端保存）
        await self.send(text_data=json.dumps({
            'type': 'connection_established',
            'data': {
                'client_id': self.client_id,
                'server_time': datetime.now().isoformat(),
                'reconnect_count': sess['reconnect_count'],
                'pending_alerts_count': len(sess['alerts'])
            }
        }))

        # 如果有暂存的告警，立即发送
        pending = self._drain_pending_alerts(self.client_id)
        for alert in pending:
            try:
                await self.send(text_data=json.dumps({
                    'type': 'alert',
                    'data': alert['data'],
                    'pending': True,
                    'queued_at': datetime.fromtimestamp(alert['queued_at']).isoformat()
                }))
            except Exception as e:
                logger.warning(f"发送暂存告警失败: {e}")

        self.is_connected = True

        # 启动心跳任务
        self.ping_task = asyncio.create_task(self._heartbeat_loop())

    async def disconnect(self, close_code):
        self.is_connected = False
        if self.ping_task:
            self.ping_task.cancel()
            try:
                await self.ping_task
            except asyncio.CancelledError:
                pass

        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

        # 清理空会话，保留有暂存告警的
        if hasattr(self, 'client_id'):
            sess = self._pending_sessions.get(self.client_id)
            if sess and len(sess['alerts']) == 0:
                # 延迟清理，给重连留窗口
                pass

    async def _heartbeat_loop(self):
        """心跳循环：定期 ping 客户端，超时则断开"""
        while self.is_connected:
            try:
                # 发送 ping
                await self.send(text_data=json.dumps({
                    'type': 'ping',
                    'data': {'timestamp': time.time()}
                }))
                self.pong_received = False

                # 等待 pong
                await asyncio.sleep(self.PONG_TIMEOUT)
                if not self.pong_received:
                    logger.warning(
                        f"客户端 {self.client_id} 心跳超时 ({self.PONG_TIMEOUT}s)，可能断线"
                    )
                    # 不强制断开，等待客户端重连
                    # 标记会话为断线状态
                    if self.client_id in self._pending_sessions:
                        self._pending_sessions[self.client_id]['last_seen'] = time.time()

                await asyncio.sleep(self.PING_INTERVAL - self.PONG_TIMEOUT)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"心跳任务异常: {e}")
                await asyncio.sleep(5)

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            msg_type = data.get('type')

            if msg_type == 'pong':
                self.pong_received = True
                if hasattr(self, 'client_id'):
                    self._track_session(self.client_id)
                return

            if msg_type == 'reconnect_request':
                # 客户端请求重连，更新会话
                old_client_id = data.get('old_client_id')
                if old_client_id and old_client_id in self._pending_sessions:
                    # 迁移暂存告警到新 client_id（如果变化了）
                    if old_client_id != self.client_id:
                        pending = self._drain_pending_alerts(old_client_id)
                        for alert in pending:
                            self._queue_pending_alert(self.client_id, alert['data'])
                        del self._pending_sessions[old_client_id]
                return

        except json.JSONDecodeError:
            logger.warning(f"无效的 WebSocket 消息: {text_data[:100]}")

    async def send_alert(self, event):
        alert_data = event['data']

        # 如果连接不稳定，暂存告警
        if not self.is_connected or not self.pong_received:
            if hasattr(self, 'client_id'):
                self._queue_pending_alert(self.client_id, alert_data)
            return

        try:
            await self.send(text_data=json.dumps({
                'type': 'alert',
                'data': alert_data
            }))
        except Exception as e:
            logger.warning(f"WebSocket 发送告警失败，暂存: {e}")
            if hasattr(self, 'client_id'):
                self._queue_pending_alert(self.client_id, alert_data)


class SpectrumConsumer(AsyncWebsocketConsumer, ReconnectingConsumerMixin):
    async def connect(self):
        self.artifact_id = self.scope['url_route']['kwargs']['artifact_id']

        query_string = self.scope.get('query_string', b'').decode()
        self.client_id = None
        for kv in query_string.split('&'):
            if kv.startswith('client_id='):
                self.client_id = kv.split('=', 1)[1]
                break
        if not self.client_id:
            import uuid
            self.client_id = 'spec_' + uuid.uuid4().hex[:12]

        self.room_group_name = f'spectrum_{self.artifact_id}'
        self.pong_received = True
        self.ping_task = None
        self.is_connected = False

        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

        sess = self._track_session(self.client_id)

        await self.send(text_data=json.dumps({
            'type': 'connection_established',
            'data': {
                'artifact_id': self.artifact_id,
                'client_id': self.client_id,
                'server_time': datetime.now().isoformat(),
                'reconnect_count': sess['reconnect_count']
            }
        }))

        self.is_connected = True
        self.ping_task = asyncio.create_task(self._heartbeat_loop())

    async def disconnect(self, close_code):
        self.is_connected = False
        if self.ping_task:
            self.ping_task.cancel()
            try:
                await self.ping_task
            except asyncio.CancelledError:
                pass

        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def _heartbeat_loop(self):
        while self.is_connected:
            try:
                await self.send(text_data=json.dumps({
                    'type': 'ping',
                    'data': {'timestamp': time.time()}
                }))
                self.pong_received = False
                await asyncio.sleep(self.PONG_TIMEOUT)
                if not self.pong_received:
                    logger.warning(
                        f"光谱客户端 {self.client_id} 心跳超时"
                    )
                await asyncio.sleep(self.PING_INTERVAL - self.PONG_TIMEOUT)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"光谱心跳异常: {e}")
                await asyncio.sleep(5)

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            if data.get('type') == 'pong':
                self.pong_received = True
                if hasattr(self, 'client_id'):
                    self._track_session(self.client_id)
        except json.JSONDecodeError:
            pass

    async def send_spectrum(self, event):
        spectrum_data = event['data']

        if not self.is_connected or not self.pong_received:
            return

        try:
            await self.send(text_data=json.dumps({
                'type': 'spectrum_update',
                'data': spectrum_data
            }))
        except Exception as e:
            logger.warning(f"WebSocket 光谱更新失败: {e}")
