import torch
import torch.nn.functional as F

G_scale = torch.tensor([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 0.]
], device="cuda", dtype=torch.float32)

G_rot = torch.tensor([
    [0., -1., 0.],
    [1.,  0., 0.],
    [0.,  0., 0.]
], device="cuda", dtype=torch.float32)

G_tx = torch.tensor([
    [0., 0., 1.],
    [0., 0., 0.],
    [0., 0., 0.]
], device="cuda", dtype=torch.float32)

G_ty = torch.tensor([
    [0., 0., 0.],
    [0., 0., 1.],
    [0., 0., 0.]
], device="cuda", dtype=torch.float32)

Lie_Basis = torch.stack([G_scale, G_rot, G_tx, G_ty], dim=0)


def apply_tensor_low_rank_filter(flow_tensor, rank=3):
    """
    Tensor-based Robust Estimation.
    """
    T, C, H, W = flow_tensor.shape
    device = flow_tensor.device
    dtype = flow_tensor.dtype
    
    matrix_mode_t = flow_tensor.reshape(T, -1).float()
    
    try:
        U, S, Vh = torch.linalg.svd(matrix_mode_t, full_matrices=False)
    except:
        print("[Warning] SVD failed, skipping tensor filter.")
        return flow_tensor

    S_low = S.clone()
    S_low[rank:] = 0
    
    matrix_reconstructed = torch.matmul(
        U * S_low.unsqueeze(0), 
        Vh
    )
    
    filtered_flow = matrix_reconstructed.reshape(T, C, H, W).to(dtype=dtype, device=device)
    return filtered_flow


def estimate_global_lie_params(flow_tensor, H, W, weights=None):
    """
    xi*
    """
    T_dim = flow_tensor.shape[0]
    device = flow_tensor.device

    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    
    coords = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).expand(T_dim, -1, -1, -1)
    coords_flat = coords.reshape(-1, 2)
    
    N_total = coords_flat.shape[0]
    J = torch.zeros(N_total, 2, 4, device=device)
    
    x = coords_flat[:, 0]
    y = coords_flat[:, 1]
    
    # Row 0 (u)
    J[:, 0, 0] = x
    J[:, 0, 1] = -y
    J[:, 0, 2] = 1.0
    
    # Row 1 (v)
    J[:, 1, 0] = y
    J[:, 1, 1] = x
    J[:, 1, 3] = 1.0
    
    scale_x = 2.0 / W
    scale_y = 2.0 / H
    
    flow_flat = flow_tensor.permute(0, 2, 3, 1).reshape(-1, 2).clone()
    flow_flat[:, 0] *= scale_x
    flow_flat[:, 1] *= scale_y
    
    if weights is None:
        A = torch.matmul(J.transpose(1, 2), J).sum(dim=0)
        b = torch.matmul(J.transpose(1, 2), flow_flat.unsqueeze(-1)).sum(dim=0)
    else:
        pass

    A_reg = A + torch.eye(4, device=device) * 1e-6
    xi_star = torch.linalg.solve(A_reg, b).squeeze()
    
    return xi_star


def apply_lie_affine_warp(phase_tensor, lie_params, basis, shape_info):
    """
    time t Sim(2) 
    """
    n_frames, H, W = shape_info
    B_rope, _, S, D = phase_tensor.shape
    
    phase_spatial = phase_tensor.view(n_frames, H, W, D).permute(0, 3, 1, 2)
    
    transform_log = torch.einsum('i,ijk->jk', lie_params, basis)
    
    device = lie_params.device
    t_scale = torch.linspace(0, 1, n_frames, device=device).view(n_frames, 1, 1)
    
    trajectory_logs = (-1.0 * t_scale) * transform_log.unsqueeze(0)

    transform_matrix = torch.linalg.matrix_exp(trajectory_logs)
    
    theta = transform_matrix[:, :2, :]
    
    grid = F.affine_grid(theta, phase_spatial.size(), align_corners=False)
    
    warped_phase = F.grid_sample(
        phase_spatial, 
        grid, 
        mode='bilinear', 
        padding_mode='reflection',
        align_corners=False
    )
    
    return warped_phase.permute(0, 2, 3, 1).reshape(B_rope, 1, S, D)