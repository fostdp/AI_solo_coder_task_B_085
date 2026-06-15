import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class JadeTexture:
    """
    玉质结构参数
    通过CT扫描标定获取
    """
    crystal_orientation: np.ndarray  # 主晶向 (3,)
    grain_size_um: float = 50.0  # 平均晶粒尺寸 (μm)
    porosity: float = 0.02  # 孔隙率
    density: float = 3.0  # 密度 (g/cm³)
    mineral_composition: Dict[str, float] = field(default_factory=lambda: {
        '透闪石': 0.85,
        '阳起石': 0.10,
        '其他': 0.05
    })
    ct_scan_id: str = ''
    scan_date: str = ''


@dataclass
class DiffusionTensor:
    """
    各向异性扩散张量 D_ij
    考虑玉质结构方向依赖性
    
    D = R · diag(D_parallel, D_perp1, D_perp2) · R^T
    
    其中 D_parallel 为沿晶向方向的扩散系数
         D_perp1, D_perp2 为垂直晶向的扩散系数
         R 为主晶向到实验室坐标系的旋转矩阵
    """
    D_parallel: float = 1.0e-14  # 平行晶向扩散系数 (m²/s)
    D_perp1: float = 5.0e-15     # 垂直晶向1扩散系数 (m²/s)
    D_perp2: float = 4.5e-15     # 垂直晶向2扩散系数 (m²/s)
    orientation_matrix: Optional[np.ndarray] = None  # 3x3 旋转矩阵 R
    principal_axes: Optional[np.ndarray] = None     # 3个主方向轴 (3,3)
    
    anisotropy_ratio: float = 0.0   # 各向异性比 D_par/D_perp_mean
    
    def __post_init__(self):
        if self.orientation_matrix is None:
            self.orientation_matrix = np.eye(3)
        if self.principal_axes is None:
            self.principal_axes = np.eye(3)
        if self.anisotropy_ratio == 0 and self.D_perp1 > 0:
            mean_perp = (self.D_perp1 + self.D_perp2) / 2
            self.anisotropy_ratio = self.D_parallel / mean_perp
    
    def get_tensor_matrix(self) -> np.ndarray:
        """
        获取完整的 3x3 扩散张量矩阵
        D = R · Λ · R^T
        """
        Lambda = np.diag([self.D_parallel, self.D_perp1, self.D_perp2])
        R = self.orientation_matrix
        return R @ Lambda @ R.T
    
    def get_effective_diffusivity(self, direction: np.ndarray) -> float:
        """
        计算沿指定方向的有效扩散系数
        D_eff = n^T · D · n
        
        Args:
            direction: 方向向量 (3,)，将自动归一化
        """
        n = np.asarray(direction, dtype=float)
        n = n / (np.linalg.norm(n) + 1e-12)
        D = self.get_tensor_matrix()
        return float(n @ D @ n)
    
    def get_surface_normal_diffusivity(
        self, 
        surface_normal: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        计算沿表面法线方向的扩散系数和主扩散方向
        
        考虑:
        1. 表面法线与晶向的夹角
        2. 表面缺陷导致的扩散增强（晶界通道效应）
        3. 缺陷辅助的短程扩散
        
        Returns:
            (D_effective, principal_direction)
        """
        n = np.asarray(surface_normal, dtype=float)
        n = n / (np.linalg.norm(n) + 1e-12)
        
        D_bulk = self.get_tensor_matrix()
        D_normal = float(n @ D_bulk @ n)
        
        angle_parallel = np.arccos(
            np.clip(n @ self.principal_axes[0], -1, 1)
        )
        grain_boundary_factor = 1.0 + 0.5 * np.sin(angle_parallel) ** 2
        
        D_gb_enhancement = 2.5
        D_effective = D_normal * grain_boundary_factor * D_gb_enhancement
        
        D_eigenvalues, D_eigenvectors = np.linalg.eigh(D_bulk)
        principal_idx = np.argmax(D_eigenvalues)
        principal_dir = D_eigenvectors[:, principal_idx]
        
        return D_effective, principal_dir


class CTCalibratedTensorBuilder:
    """
    基于CT扫描数据的扩散张量标定器
    
    标定流程:
    1. 从CT切片提取晶粒取向（EBSD类似方法）
    2. 计算孔隙网络连通性张量
    3. 关联晶向与扩散主方向
    4. 标定各向异性系数
    """
    
    def __init__(self):
        self.calibration_cache = {}
    
    def build_from_ct_scan(
        self,
        ct_volume: np.ndarray,  # 3D CT体数据 (H, W, D)
        voxel_size_um: float = 50.0,
        jade_type: str = '和田玉'
    ) -> DiffusionTensor:
        """
        从CT体数据构建扩散张量
        
        Args:
            ct_volume: 3D灰度体数据
            voxel_size_um: 体素尺寸 (μm)
            jade_type: 玉种类型
        """
        cache_key = f"{jade_type}_{ct_volume.shape}_{voxel_size_um}"
        if cache_key in self.calibration_cache:
            return self.calibration_cache[cache_key].copy()
        
        orientation = self._extract_crystal_orientation(ct_volume)
        R, principal_axes = self._build_rotation_matrix(orientation)
        
        base_params = self._get_base_params(jade_type)
        
        porosity = self._calculate_porosity(ct_volume)
        pore_factor = self._pore_enhancement_factor(porosity)
        
        grain_size = self._estimate_grain_size(ct_volume, voxel_size_um)
        gb_factor = self._grain_boundary_factor(grain_size)
        
        D_parallel = base_params['D0_parallel'] * pore_factor * gb_factor
        D_perp1 = base_params['D0_perp1'] * pore_factor * gb_factor * 0.5
        D_perp2 = base_params['D0_perp2'] * pore_factor * gb_factor * 0.45
        
        tensor = DiffusionTensor(
            D_parallel=D_parallel,
            D_perp1=D_perp1,
            D_perp2=D_perp2,
            orientation_matrix=R,
            principal_axes=principal_axes
        )
        
        self.calibration_cache[cache_key] = tensor
        return tensor
    
    def build_preset(
        self,
        jade_culture: str = '红山文化',
        jade_type: str = '玉璧',
        orientation_deg: Tuple[float, float, float] = (0, 0, 0)
    ) -> DiffusionTensor:
        """
        基于考古先验知识构建预设扩散张量（无CT数据时使用）
        
        Args:
            jade_culture: 文化类型 (红山/良渚)
            jade_type: 器型 (玉璧/玉琮/...)
            orientation_deg: 主晶向欧拉角 (α, β, γ) 度
        """
        culture_params = {
            '红山文化': {
                'D0_parallel': 5.0e-14,
                'D0_perp1': 2.2e-14,
                'D0_perp2': 2.0e-14,
                'texture_factor': 1.2
            },
            '良渚文化': {
                'D0_parallel': 3.8e-14,
                'D0_perp1': 1.8e-14,
                'D0_perp2': 1.6e-14,
                'texture_factor': 1.1
            }
        }
        
        params = culture_params.get(jade_culture, culture_params['红山文化'])
        
        type_orientation = {
            '玉璧': (0, 0, 0),
            '玉琮': (90, 0, 0),
            '玉钺': (45, 30, 15),
            '玉璜': (0, 60, 0),
            '玉珠': (30, 45, 60),
            '玉管': (90, 0, 0),
            '玉兽': (20, 30, 45),
            '玉鸟': (45, 15, 30)
        }
        
        base_orient = type_orientation.get(jade_type, (0, 0, 0))
        final_orient = tuple(
            base_orient[i] + orientation_deg[i] for i in range(3)
        )
        
        R, principal_axes = self._euler_to_rotation_matrix(final_orient)
        
        return DiffusionTensor(
            D_parallel=params['D0_parallel'],
            D_perp1=params['D0_perp1'],
            D_perp2=params['D0_perp2'],
            orientation_matrix=R,
            principal_axes=principal_axes
        )
    
    def _extract_crystal_orientation(self, ct_volume: np.ndarray) -> np.ndarray:
        """
        从CT数据提取主晶向
        方法: 结构张量的主特征向量
        """
        grad_x = np.gradient(ct_volume.astype(float), axis=0)
        grad_y = np.gradient(ct_volume.astype(float), axis=1)
        grad_z = np.gradient(ct_volume.astype(float), axis=2)
        
        S = np.zeros((3, 3))
        S[0, 0] = np.mean(grad_x ** 2)
        S[1, 1] = np.mean(grad_y ** 2)
        S[2, 2] = np.mean(grad_z ** 2)
        S[0, 1] = S[1, 0] = np.mean(grad_x * grad_y)
        S[0, 2] = S[2, 0] = np.mean(grad_x * grad_z)
        S[1, 2] = S[2, 1] = np.mean(grad_y * grad_z)
        
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        return eigenvectors[:, np.argmax(eigenvalues)]
    
    def _build_rotation_matrix(
        self, 
        principal_direction: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据主方向构建完整旋转矩阵
        """
        n = principal_direction / (np.linalg.norm(principal_direction) + 1e-12)
        
        if abs(n[0]) < 0.9:
            e2 = np.array([0, n[2], -n[1]])
        else:
            e2 = np.array([-n[1], n[0], 0])
        e2 = e2 / (np.linalg.norm(e2) + 1e-12)
        
        e3 = np.cross(n, e2)
        
        R = np.column_stack([n, e2, e3])
        principal_axes = R.copy()
        return R, principal_axes
    
    def _euler_to_rotation_matrix(
        self, 
        euler_deg: Tuple[float, float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ZXZ欧拉角转旋转矩阵
        """
        alpha, beta, gamma = np.radians(euler_deg)
        
        Rz1 = np.array([
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]
        ])
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(beta), -np.sin(beta)],
            [0, np.sin(beta), np.cos(beta)]
        ])
        
        Rz2 = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])
        
        R = Rz1 @ Rx @ Rz2
        principal_axes = R.copy()
        return R, principal_axes
    
    def _calculate_porosity(self, ct_volume: np.ndarray) -> float:
        """计算孔隙率"""
        threshold = np.percentile(ct_volume, 10)
        pore_voxels = np.sum(ct_volume < threshold)
        total_voxels = ct_volume.size
        return float(pore_voxels / total_voxels)
    
    def _pore_enhancement_factor(self, porosity: float) -> float:
        """孔隙对扩散的增强因子"""
        return 1.0 + 3.0 * porosity / (1.0 - porosity + 1e-6)
    
    def _estimate_grain_size(
        self, ct_volume: np.ndarray, voxel_size_um: float
    ) -> float:
        """估算平均晶粒尺寸"""
        sample = ct_volume[:100, :100, :100]
        edges = np.abs(np.gradient(sample.astype(float)))
        edge_strength = np.sum(edges, axis=0)
        threshold = np.percentile(edge_strength, 90)
        edge_map = edge_strength > threshold
        
        from scipy import ndimage as ndi
        labeled, n_grains = ndi.label(edge_map == 0)
        if n_grains > 0:
            sizes = ndi.sum(np.ones_like(labeled), labeled, 
                          index=np.arange(1, n_grains + 1))
            mean_size_voxels = np.mean(sizes)
            grain_size = (mean_size_voxels ** (1/3)) * voxel_size_um
        else:
            grain_size = 50.0
        
        return float(grain_size)
    
    def _grain_boundary_factor(self, grain_size_um: float) -> float:
        """晶界扩散增强因子"""
        D_lattice = 1.0
        D_grain_boundary = 1000.0
        d = grain_size_um
        
        fraction_gb = min(0.5, 2.0 / d)
        return (1 - fraction_gb) * D_lattice + fraction_gb * D_grain_boundary
    
    def _get_base_params(self, jade_type: str) -> Dict[str, float]:
        """获取不同玉种的基础扩散参数"""
        params_dict = {
            '和田玉': {'D0_parallel': 5.0e-14, 'D0_perp1': 2.0e-14, 'D0_perp2': 1.8e-14},
            '岫岩玉': {'D0_parallel': 6.5e-14, 'D0_perp1': 2.8e-14, 'D0_perp2': 2.5e-14},
            '蓝田玉': {'D0_parallel': 5.8e-14, 'D0_perp1': 2.5e-14, 'D0_perp2': 2.2e-14},
            '独山玉': {'D0_parallel': 4.2e-14, 'D0_perp1': 1.9e-14, 'D0_perp2': 1.7e-14}
        }
        return params_dict.get(jade_type, params_dict['和田玉'])


class AnisotropicDiffusionSolver:
    """
    各向异性扩散方程有限差分求解器
    
    求解: ∂C/∂t = ∇·(D·∇C)
    
    其中 D 为空间依赖的扩散张量
    使用算子分裂 + 交替方向隐式 (ADI) 方法
    """
    
    def __init__(
        self,
        tensor: DiffusionTensor,
        grid_size_mm: Tuple[int, int, int] = (50, 50, 50),
        grid_spacing_mm: float = 0.1
    ):
        self.tensor = tensor
        self.grid_size = grid_size_mm
        self.dx = grid_spacing_mm / 1000  # 转换为米
        self.D_tensor = tensor.get_tensor_matrix()
        
        self.C = np.zeros(grid_size_mm)
        self.boundary_mask = np.zeros(grid_size_mm, dtype=bool)
        
        self._precompute_direction_diffusivities()
    
    def _precompute_direction_diffusivities(self):
        """预计算6个方向的有效扩散系数"""
        directions = {
            'x+': np.array([1, 0, 0]),
            'x-': np.array([-1, 0, 0]),
            'y+': np.array([0, 1, 0]),
            'y-': np.array([0, -1, 0]),
            'z+': np.array([0, 0, 1]),
            'z-': np.array([0, 0, -1])
        }
        
        self.D_directions = {}
        for name, d in directions.items():
            self.D_directions[name] = self.tensor.get_effective_diffusivity(d)
    
    def set_boundary_condition(
        self,
        surface_concentration: float,
        surface_normal: np.ndarray = np.array([0, 0, 1])
    ):
        """设置表面边界条件"""
        self.C[0, :, :] = surface_concentration
        self.boundary_mask[0, :, :] = True
        
        D_surface, _ = self.tensor.get_surface_normal_diffusivity(surface_normal)
        self.D_directions['z+'] = D_surface
        self.D_directions['z-'] = D_surface * 0.5
    
    def step(self, dt_s: float, n_steps: int = 1) -> np.ndarray:
        """
        执行 n_steps 步时间推进
        
        Args:
            dt_s: 时间步长 (秒)
            n_steps: 步数
        """
        for _ in range(n_steps):
            self._adi_step(dt_s)
        return self.C.copy()
    
    def _adi_step(self, dt_s: float):
        """
        交替方向隐式单步推进
        
        分三步: x方向隐式, y方向隐式, z方向隐式
        """
        Dx = (self.D_directions['x+'] + self.D_directions['x-']) / 2
        Dy = (self.D_directions['y+'] + self.D_directions['y-']) / 2
        Dz = (self.D_directions['z+'] + self.D_directions['z-']) / 2
        
        rx = Dx * dt_s / (self.dx ** 2)
        ry = Dy * dt_s / (self.dx ** 2)
        rz = Dz * dt_s / (self.dx ** 2)
        
        C_new = self.C.copy()
        
        C_new = self._tridiagonal_solve_1d(C_new, rx, axis=0)
        C_new = self._tridiagonal_solve_1d(C_new, ry, axis=1)
        C_new = self._tridiagonal_solve_1d(C_new, rz, axis=2)
        
        self.C[0, :, :] = self.C[0, :, :]  # 保持边界条件
        self.C = C_new
    
    def _tridiagonal_solve_1d(
        self, 
        C: np.ndarray, 
        r: float, 
        axis: int
    ) -> np.ndarray:
        """
        1D三对角系统求解 (Thomas算法)
        
        方程: -r C_{i-1} + (1+2r) C_i - r C_{i+1} = C_i^old
        """
        C = np.moveaxis(C, axis, 0)
        n = C.shape[0]
        
        a = -r * np.ones(n - 1)
        b = (1 + 2 * r) * np.ones(n)
        c = -r * np.ones(n - 1)
        
        b[0] = 1.0
        c[0] = 0.0
        a[-1] = 0.0
        b[-1] = 1.0
        
        result = np.zeros_like(C)
        shape = C.shape[1:]
        
        idx = [slice(None)] + [0] * len(shape)
        for flat_idx in range(np.prod(shape)):
            multi_idx = np.unravel_index(flat_idx, shape)
            full_idx = (slice(None),) + tuple(multi_idx)
            
            d = C[full_idx].copy()
            
            cp = c.copy()
            dp = d.copy()
            dp[0] = d[0]
            
            for i in range(1, n):
                m = a[i-1] / b[i-1]
                b[i] -= m * c[i-1]
                dp[i] -= m * dp[i-1]
            
            x = np.zeros(n)
            x[-1] = dp[-1] / b[-1]
            for i in range(n-2, -1, -1):
                x[i] = (dp[i] - c[i] * x[i+1]) / b[i]
            
            result[full_idx] = x
        
        return np.moveaxis(result, 0, axis)
    
    def get_depth_profile(self, axis: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定轴的平均浓度深度分布
        
        Returns:
            (depth_mm, concentration)
        """
        axes = tuple(i for i in range(3) if i != axis)
        profile = np.mean(self.C, axis=axes)
        depths = np.arange(len(profile)) * self.dx * 1000
        return depths, profile
    
    def get_cross_section(self, pos_ratio: float = 0.5, axis: int = 0) -> np.ndarray:
        """获取指定位置的截面浓度分布"""
        idx = int(pos_ratio * (self.grid_size[axis] - 1))
        slicing = [slice(None)] * 3
        slicing[axis] = idx
        return self.C[tuple(slicing)].copy()


_tensor_builder = CTCalibratedTensorBuilder()

def get_default_tensor(jade_culture: str, jade_type: str) -> DiffusionTensor:
    """获取默认的预设扩散张量"""
    return _tensor_builder.build_preset(jade_culture, jade_type)
