import threading
import time
import random
import numpy as np
from datetime import datetime
import logging
import requests
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class NetworkQoSConfig:
    """5G 网络 QoS 配置"""
    bandwidth_mhz: float = 100.0
    peak_throughput_gbps: float = 1.0
    sustained_throughput_gbps: float = 0.8
    latency_ms: float = 10.0
    jitter_ms: float = 2.0
    packet_loss_rate: float = 0.001
    max_queue_packets: int = 1000


@dataclass
class NetworkStats:
    """5G 网络统计"""
    total_bytes_sent: int = 0
    total_packets_sent: int = 0
    total_packets_dropped: int = 0
    avg_latency_ms: float = 0.0
    current_throughput_mbps: float = 0.0
    queue_occupancy: int = 0
    congestion_level: float = 0.0


class TokenBucket:
    """令牌桶算法 - 5G 带宽限制核心"""

    def __init__(self, rate_bits_per_sec: float, bucket_size_bits: float):
        """
        :param rate_bits_per_sec: 令牌生成速率（bit/s）
        :param bucket_size_bits: 桶容量（最大突发流量）
        """
        self.rate = rate_bits_per_sec
        self.bucket_size = bucket_size_bits
        self.current_tokens = bucket_size_bits
        self.last_update = time.time()
        self._lock = threading.Lock()

    def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_update
        if elapsed > 0:
            self.current_tokens = min(
                self.bucket_size,
                self.current_tokens + elapsed * self.rate
            )
            self.last_update = now

    def consume(self, bits: float, block: bool = True, timeout: float = 5.0) -> bool:
        """
        消耗令牌
        :param bits: 需要消耗的位数
        :param block: 是否阻塞等待
        :param timeout: 超时时间（秒）
        :return: 是否成功消耗
        """
        start_time = time.time()
        with self._lock:
            while True:
                self._refill()
                if self.current_tokens >= bits:
                    self.current_tokens -= bits
                    return True

                if not block:
                    return False

                wait_time = (bits - self.current_tokens) / self.rate
                wait_time = min(wait_time, timeout - (time.time() - start_time))

                if wait_time <= 0:
                    return False

                time.sleep(wait_time)

                if time.time() - start_time >= timeout:
                    return False

    def available_tokens(self) -> float:
        """获取当前可用令牌"""
        with self._lock:
            self._refill()
            return self.current_tokens

    def set_rate(self, new_rate_bits_per_sec: float):
        """动态调整速率（用于网络拥塞控制）"""
        with self._lock:
            self._refill()
            self.rate = new_rate_bits_per_sec


class FiveGNetworkSimulator:
    """
    5G NR 网络模拟器 - 修复 simulator/5g.py 带宽限制技术债
    
    核心特性：
    1. 令牌桶带宽限制（可配置 5G 频段和带宽）
    2. 5G QoS 流模拟（eMBB、URLLC、mMTC）
    3. 包调度和队列管理
    4. 网络拥塞控制（RED 主动队列管理）
    5. 实时流量统计
    """

    # 5G 频段配置（部分典型值）
    BAND_CONFIGS = {
        'n78': {'bandwidth': 100, 'peak_gbps': 1.0, 'freq': '3.5GHz'},
        'n41': {'bandwidth': 100, 'peak_gbps': 0.8, 'freq': '2.6GHz'},
        'n77': {'bandwidth': 100, 'peak_gbps': 1.2, 'freq': '3.3GHz'},
        'n257': {'bandwidth': 400, 'peak_gbps': 5.0, 'freq': '28GHz'},
        'n79': {'bandwidth': 100, 'peak_gbps': 0.9, 'freq': '4.9GHz'},
    }

    # QoS 流类型
    QOS_FLOWS = {
        'embb': {'priority': 2, 'latency_budget_ms': 100, 'name': '增强移动宽带'},
        'urllc': {'priority': 1, 'latency_budget_ms': 10, 'name': '超可靠低时延'},
        'mmtc': {'priority': 5, 'latency_budget_ms': 1000, 'name': '海量机器类通信'},
    }

    def __init__(self, band: str = 'n78', qos_config: Optional[NetworkQoSConfig] = None):
        self.band = band
        band_cfg = self.BAND_CONFIGS.get(band, self.BAND_CONFIGS['n78'])

        if qos_config is None:
            qos_config = NetworkQoSConfig(
                bandwidth_mhz=band_cfg['bandwidth'],
                peak_throughput_gbps=band_cfg['peak_gbps']
            )

        self.qos_config = qos_config

        # 计算实际带宽：100MHz 5G NR 子载波=1200，符号率~15kHz，约 1Gbps 峰值
        peak_bps = qos_config.peak_throughput_gbps * 1e9
        sustained_bps = qos_config.sustained_throughput_gbps * 1e9

        # 令牌桶：持续速率 = 持续吞吐量，突发容量 = 峰值 * 100ms
        self.token_bucket = TokenBucket(
            rate_bits_per_sec=sustained_bps,
            bucket_size_bits=peak_bps * 0.1
        )

        # 包队列
        self._queue: Deque[Dict] = deque()
        self._queue_lock = threading.Lock()

        # 统计
        self.stats = NetworkStats()
        self._history: Deque[tuple] = deque(maxlen=100)
        self._last_throughput_calc = time.time()
        self._bytes_in_window = 0

        # 拥塞控制
        self._congestion_control_enabled = True
        self._current_rate_multiplier = 1.0

        # 调度线程
        self._scheduler_running = False
        self._scheduler_thread: Optional[threading.Thread] = None

    def start(self):
        """启动网络调度器"""
        if self._scheduler_running:
            return
        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name='5G-Scheduler'
        )
        self._scheduler_thread.start()
        logger.info(
            f"5G 网络模拟器已启动，频段: {self.band} "
            f"({self.BAND_CONFIGS[self.band]['freq']}), "
            f"带宽: {self.qos_config.bandwidth_mhz}MHz, "
            f"峰值: {self.qos_config.peak_throughput_gbps}Gbps"
        )

    def stop(self):
        """停止网络调度器"""
        self._scheduler_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=2)
        logger.info("5G 网络模拟器已停止")

    def _scheduler_loop(self):
        """调度循环 - 模拟 MAC 层调度"""
        while self._scheduler_running:
            try:
                self._process_queue()
                self._update_stats()
                self._congestion_control()
                time.sleep(0.001)  # 1ms 调度间隔（5G TTI = 1ms）
            except Exception as e:
                logger.error(f"5G 调度器错误: {e}")
                time.sleep(0.01)

    def _process_queue(self):
        """处理发送队列（轮询调度）"""
        with self._queue_lock:
            if not self._queue:
                self.stats.queue_occupancy = 0
                return

            self.stats.queue_occupancy = len(self._queue)
            packet = self._queue.popleft()

        # 模拟传输延迟
        latency = (
            self.qos_config.latency_ms
            + np.random.normal(0, self.qos_config.jitter_ms)
        )
        latency = max(0.1, latency)

        # 模拟包丢失
        if random.random() < self.qos_config.packet_loss_rate:
            self.stats.total_packets_dropped += 1
            if packet.get('callback'):
                try:
                    packet['callback'](success=False, reason='packet_loss')
                except:
                    pass
            return

        # 带宽限制：等待令牌
        packet_bits = packet['size_bytes'] * 8
        got_tokens = self.token_bucket.consume(
            packet_bits,
            block=True,
            timeout=5.0
        )

        if not got_tokens:
            self.stats.total_packets_dropped += 1
            if packet.get('callback'):
                try:
                    packet['callback'](success=False, reason='timeout')
                except:
                    pass
            return

        # 模拟传输时间
        time.sleep(latency / 1000.0)

        # 更新统计
        self.stats.total_bytes_sent += packet['size_bytes']
        self.stats.total_packets_sent += 1
        self._bytes_in_window += packet['size_bytes']

        # 计算平均延迟
        self.stats.avg_latency_ms = (
            self.stats.avg_latency_ms * 0.95 + latency * 0.05
        )

        # 回调
        if packet.get('callback'):
            try:
                packet['callback'](
                    success=True,
                    latency_ms=latency,
                    throughput_mbps=(packet_bits / (latency / 1000)) / 1e6
                )
            except:
                pass

    def _update_stats(self):
        """更新吞吐量统计"""
        now = time.time()
        window = now - self._last_throughput_calc
        if window >= 1.0:
            self.stats.current_throughput_mbps = (
                self._bytes_in_window * 8 / window / 1e6
            )
            self._bytes_in_window = 0
            self._last_throughput_calc = now

            self._history.append((now, self.stats.current_throughput_mbps))

            queue_len = len(self._queue)
            self.stats.congestion_level = min(
                1.0,
                queue_len / self.qos_config.max_queue_packets
            )

    def _congestion_control(self):
        """RED 主动队列管理 + 速率自适应"""
        if not self._congestion_control_enabled:
            return

        queue_len = len(self._queue)
        max_queue = self.qos_config.max_queue_packets

        # RED 算法：队列长度超过阈值时随机丢包
        if queue_len > max_queue * 0.5:
            drop_prob = min(
                0.5,
                (queue_len - max_queue * 0.5) / (max_queue * 0.5)
            )
            if random.random() < drop_prob * 0.01:
                with self._queue_lock:
                    if self._queue:
                        self._queue.popleft()
                        self.stats.total_packets_dropped += 1

        # 速率自适应：根据拥塞程度调整令牌桶速率
        target_multiplier = max(0.3, 1.0 - self.stats.congestion_level * 0.7)
        self._current_rate_multiplier = (
            self._current_rate_multiplier * 0.95 + target_multiplier * 0.05
        )

        sustained_bps = (
            self.qos_config.sustained_throughput_gbps
            * 1e9
            * self._current_rate_multiplier
        )
        self.token_bucket.set_rate(sustained_bps)

    def send_packet(
        self,
        data_bytes: int,
        qos_flow: str = 'embb',
        priority: Optional[int] = None,
        callback: Optional[callable] = None
    ) -> bool:
        """
        发送数据包（放入发送队列）
        :param data_bytes: 数据大小（字节）
        :param qos_flow: QoS 流类型 (embb/urllc/mmtc)
        :param priority: 优先级（1-5，越小越高）
        :param callback: 发送完成回调
        :return: 是否成功入队
        """
        if len(self._queue) >= self.qos_config.max_queue_packets:
            # 队列满，主动丢包
            self.stats.total_packets_dropped += 1
            return False

        if priority is None:
            priority = self.QOS_FLOWS.get(qos_flow, {}).get('priority', 3)

        packet = {
            'size_bytes': data_bytes,
            'qos_flow': qos_flow,
            'priority': priority,
            'timestamp': time.time(),
            'callback': callback
        }

        with self._queue_lock:
            # 高优先级插入队头
            if priority <= 2:
                self._queue.appendleft(packet)
            else:
                self._queue.append(packet)

        return True

    def wait_for_drain(self, timeout: float = 30.0) -> bool:
        """等待队列排空"""
        start = time.time()
        while len(self._queue) > 0:
            if time.time() - start >= timeout:
                return False
            time.sleep(0.01)
        return True

    def get_stats(self) -> Dict:
        """获取网络统计"""
        return {
            'band': self.band,
            'bandwidth_mhz': self.qos_config.bandwidth_mhz,
            'peak_throughput_gbps': self.qos_config.peak_throughput_gbps,
            **self.stats.__dict__,
            'queue_size': len(self._queue),
            'rate_multiplier': self._current_rate_multiplier,
            'available_tokens_mb': self.token_bucket.available_tokens() / 8 / 1e6,
            'history_mbps': [
                {'time': t.isoformat(), 'mbps': m}
                for t, m in self._history
            ]
        }


class Jade5GSimulator:
    """
    5G数据模拟器（带宽限制修复版）
    模拟拉曼光谱仪和X射线荧光光谱仪每6小时上报数据
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        band: str = 'n78',
        enable_network_sim: bool = True
    ):
        self.base_url = base_url
        self.is_running = False
        self.thread = None
        self.interval = 30
        self.artifact_count = 200
        self.raman_devices = [f"RAMAN{str(i).zfill(3)}" for i in range(1, 21)]
        self.xrf_devices = [f"XRF{str(i).zfill(3)}" for i in range(1, 21)]
        
        self.base_spectra_cache = {}

        # 5G 网络模拟器
        self.enable_network_sim = enable_network_sim
        self.network_sim: Optional[FiveGNetworkSimulator] = None
        if enable_network_sim:
            self.network_sim = FiveGNetworkSimulator(band=band)
            self.network_sim.start()

        # 统计
        self._upload_stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'dropped': 0
        }

    def start(self, interval: int = 30):
        """
        启动模拟器
        
        Args:
            interval: 上报间隔（秒），默认30秒模拟6小时
        """
        if self.is_running:
            logger.warning("模拟器已在运行")
            return
        
        self.interval = interval
        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        if self.enable_network_sim and not self.network_sim:
            self.network_sim = FiveGNetworkSimulator()
            self.network_sim.start()

        logger.info(
            f"5G数据模拟器已启动，上报间隔: {interval}秒，"
            f"网络模拟: {'启用' if self.enable_network_sim else '禁用'}"
        )
    
    def stop(self):
        """停止模拟器"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.network_sim:
            self.network_sim.stop()
        logger.info(
            f"5G数据模拟器已停止，统计: {self._upload_stats}, "
            f"网络: {self.network_sim.get_stats() if self.network_sim else 'N/A'}"
        )
    
    def _run(self):
        """模拟器主循环"""
        while self.is_running:
            try:
                self._generate_batch_data()
            except Exception as e:
                logger.error(f"模拟器运行出错: {e}")
            
            time.sleep(self.interval)
    
    def _generate_batch_data(self):
        """生成一批数据（模拟所有设备的一次上报）"""
        logger.info(f"开始生成光谱数据，时间: {datetime.now()}")
        
        artifact_ids = [f"JD{str(i).zfill(4)}" for i in range(1, self.artifact_count + 1)]
        
        for i, artifact_id in enumerate(artifact_ids):
            try:
                raman_device = self.raman_devices[i % len(self.raman_devices)]
                xrf_device = self.xrf_devices[i % len(self.xrf_devices)]
                
                raman_data = self._generate_raman_spectrum(artifact_id)
                xrf_data = self._generate_xrf_spectrum(artifact_id)
                
                self._upload_spectrum(artifact_id, raman_device, 'raman', raman_data)
                self._upload_spectrum(artifact_id, xrf_device, 'xrf', xrf_data)
                
            except Exception as e:
                logger.error(f"生成玉器 {artifact_id} 数据失败: {e}")

        # 等待网络队列排空
        if self.network_sim:
            drained = self.network_sim.wait_for_drain(timeout=30)
            if not drained:
                logger.warning("网络队列未在超时时间内排空")
        
        logger.info(
            f"完成 {self.artifact_count} 件玉器的光谱数据生成，"
            f"统计: {self._upload_stats}, "
            f"网络拥塞: {self.network_sim.stats.congestion_level:.1%}"
            if self.network_sim else ""
        )
    
    def _estimate_packet_size(self, spectrum_data: dict) -> int:
        """估算数据包大小（字节）"""
        import json
        try:
            payload = {
                'artifact_id': 'test',
                'device_id': 'test',
                'type': 'test',
                'spectrum_data': spectrum_data.get('spectrum_data', []),
                'wavelengths': spectrum_data.get('wavelengths', []),
                'energies': spectrum_data.get('energies', [])
            }
            return len(json.dumps(payload).encode('utf-8'))
        except:
            return 8192

    def _upload_spectrum(self, artifact_id: str, device_id: str, 
                         spectrum_type: str, spectrum_data: dict):
        """
        上传光谱数据到后端（经过 5G 网络模拟器）
        
        Args:
            artifact_id: 玉器ID
            device_id: 设备ID
            spectrum_type: 光谱类型 ('raman' 或 'xrf')
            spectrum_data: 光谱数据
        """
        self._upload_stats['total'] += 1

        try:
            payload = {
                'artifact_id': artifact_id,
                'device_id': device_id,
                'type': spectrum_type,
                'spectrum_data': spectrum_data['spectrum_data'],
                'wavelengths': spectrum_data.get('wavelengths', []),
                'energies': spectrum_data.get('energies', [])
            }

            # 通过 5G 网络模拟器
            if self.network_sim:
                packet_size = self._estimate_packet_size(spectrum_data)

                upload_completed = threading.Event()
                upload_result = {'success': False}

                def _network_callback(success, **kwargs):
                    upload_result['success'] = success
                    upload_result.update(kwargs)
                    upload_completed.set()

                queued = self.network_sim.send_packet(
                    data_bytes=packet_size,
                    qos_flow='embb',
                    callback=_network_callback
                )

                if not queued:
                    self._upload_stats['dropped'] += 1
                    logger.debug(
                        f"网络队列已满，丢包: {artifact_id} - {spectrum_type}"
                    )
                    return

                # 等待网络传输完成
                if not upload_completed.wait(timeout=30):
                    self._upload_stats['failed'] += 1
                    logger.debug(
                        f"网络传输超时: {artifact_id} - {spectrum_type}"
                    )
                    return

                if not upload_result['success']:
                    self._upload_stats['failed'] += 1
                    logger.debug(
                        f"网络传输失败: {artifact_id} - {spectrum_type}, "
                        f"原因: {upload_result.get('reason', 'unknown')}"
                    )
                    return

            # 实际 HTTP 上传
            url = f"{self.base_url}/api/spectrum/upload/"
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self._upload_stats['success'] += 1
                logger.debug(f"上传成功: {artifact_id} - {spectrum_type}")
            else:
                self._upload_stats['failed'] += 1
                logger.warning(
                    f"上传失败: {artifact_id} - {spectrum_type} - {response.status_code}"
                )
                
        except requests.exceptions.RequestException as e:
            self._upload_stats['failed'] += 1
            logger.debug(f"上传连接失败: {e}")
    
    def _generate_raman_spectrum(self, artifact_id: str) -> dict:
        """
        生成拉曼光谱数据
        
        拉曼光谱典型特征:
        - 硅氧键振动: ~500 cm⁻¹
        - 金属氧键: 100-300 cm⁻¹
        - 碳酸盐: ~1080 cm⁻¹
        """
        num_points = 512
        wavelengths = np.linspace(100, 2000, num_points)
        
        if artifact_id not in self.base_spectra_cache:
            base_intensity = np.zeros(num_points)
            
            self._add_peak(base_intensity, wavelengths, 200, 1.0, 30)
            self._add_peak(base_intensity, wavelengths, 380, 0.8, 40)
            self._add_peak(base_intensity, wavelengths, 520, 1.5, 50)
            self._add_peak(base_intensity, wavelengths, 700, 0.6, 35)
            self._add_peak(base_intensity, wavelengths, 1080, 0.9, 45)
            
            base_intensity += np.random.normal(0, 0.02, num_points)
            self.base_spectra_cache[artifact_id] = base_intensity.copy()
        
        intensity = self.base_spectra_cache[artifact_id].copy()
        
        drift = np.random.normal(0, 0.01, num_points)
        intensity += drift
        
        noise_level = 0.03
        noise = np.random.normal(0, noise_level, num_points)
        intensity += noise
        
        intensity = np.maximum(intensity, 0)
        
        return {
            'wavelengths': wavelengths.tolist(),
            'spectrum_data': intensity.tolist(),
            'laser_wavelength': 532,
            'exposure_time': 10,
            'accumulations': 3
        }
    
    def _generate_xrf_spectrum(self, artifact_id: str) -> dict:
        """
        生成X射线荧光光谱数据
        
        主要元素特征峰:
        - Si Kα: 1.74 keV
        - O Kα: 0.525 keV
        - Fe Kα: 6.40 keV
        - Ca Kα: 3.69 keV
        - Al Kα: 1.49 keV
        - Mn Kα: 5.90 keV
        - Cu Kα: 8.04 keV
        """
        num_points = 256
        energies = np.linspace(0, 15, num_points)
        
        if artifact_id not in self.base_spectra_cache:
            self.base_spectra_cache[artifact_id] = {}
        
        cache_key = 'xrf_' + artifact_id
        if cache_key not in self.base_spectra_cache:
            base_intensity = np.zeros(num_points)
            
            self._add_peak(base_intensity, energies, 0.525, 1.2, 0.05)
            self._add_peak(base_intensity, energies, 1.49, 0.8, 0.04)
            self._add_peak(base_intensity, energies, 1.74, 1.5, 0.06)
            self._add_peak(base_intensity, energies, 3.69, 0.6, 0.05)
            self._add_peak(base_intensity, energies, 6.40, 0.4, 0.04)
            self._add_peak(base_intensity, energies, 5.90, 0.15, 0.03)
            self._add_peak(base_intensity, energies, 8.04, 0.1, 0.02)
            
            background = 0.05 * np.exp(-energies / 2)
            base_intensity += background
            
            base_intensity += np.random.normal(0, 0.005, num_points)
            self.base_spectra_cache[cache_key] = base_intensity.copy()
        
        intensity = self.base_spectra_cache[cache_key].copy()
        
        drift_factor = 1 + np.random.normal(0, 0.02)
        intensity *= drift_factor
        
        noise = np.random.normal(0, 0.01, num_points)
        intensity += noise
        intensity = np.maximum(intensity, 0)
        
        is_suspect = hash(artifact_id) % 100 < 15
        if is_suspect:
            fe_peak_idx = np.argmin(np.abs(energies - 6.40))
            intensity[fe_peak_idx-5:fe_peak_idx+5] *= 2.5
            
            cu_peak_idx = np.argmin(np.abs(energies - 8.04))
            intensity[cu_peak_idx-3:cu_peak_idx+3] *= 3.0
        
        return {
            'energies': energies.tolist(),
            'spectrum_data': intensity.tolist(),
            'tube_voltage': 40,
            'tube_current': 100,
            'measurement_time': 30
        }
    
    def _add_peak(self, spectrum: np.ndarray, x_axis: np.ndarray, 
                  center: float, height: float, width: float):
        """向光谱添加高斯峰"""
        peak = height * np.exp(-((x_axis - center) ** 2) / (2 * width ** 2))
        spectrum += peak

    def get_network_stats(self) -> Optional[Dict]:
        """获取 5G 网络统计信息"""
        if self.network_sim:
            return self.network_sim.get_stats()
        return None

    def get_upload_stats(self) -> Dict:
        """获取上传统计"""
        stats = dict(self._upload_stats)
        if self._upload_stats['total'] > 0:
            stats['success_rate'] = (
                self._upload_stats['success'] / self._upload_stats['total']
            )
        return stats
    
    def generate_single(self, artifact_id: str) -> dict:
        """生成单个玉器的完整数据"""
        raman_device = self.raman_devices[hash(artifact_id) % len(self.raman_devices)]
        xrf_device = self.xrf_devices[hash(artifact_id) % len(self.xrf_devices)]
        
        return {
            'artifact_id': artifact_id,
            'timestamp': datetime.now().isoformat(),
            'raman': {
                'device_id': raman_device,
                'data': self._generate_raman_spectrum(artifact_id)
            },
            'xrf': {
                'device_id': xrf_device,
                'data': self._generate_xrf_spectrum(artifact_id)
            }
        }


simulator = Jade5GSimulator()

