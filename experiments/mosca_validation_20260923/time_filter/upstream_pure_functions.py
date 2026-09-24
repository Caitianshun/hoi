# Exact AST-extracted upstream functions, CPU globals supplied by caller.
def line_segment_init(track_mask, point_ref):
    logging.info("Naive Line Segment Init")
    # ! this function is a bad init, but this doesn't matter, later will directly optimize the curve

    tracl_mask_valid_cnt = track_mask.sum(0)
    working_mask = tracl_mask_valid_cnt > 0
    logging.info(f"Line Segment Init, invalid curve cnt={(~working_mask).sum()}")

    working_point_ref = point_ref.detach().clone()[:, working_mask]
    working_track_mask = track_mask[:, working_mask]

    T, N = track_mask.shape
    # point_ref # T,N,3
    # scan the T, for each empty slot, identify the right ends, and compute linear interpolation, if there is only one side, stay at the same position, if two end are empty, assert error, there shouldn't be an empty noodle!
    inverse_muti = torch.Tensor([i + 1 for i in range(T)][::-1]).to(working_point_ref)
    for t in tqdm(range(T)):
        to_fill_mask = ~working_track_mask[t]
        if not to_fill_mask.any():
            continue  # skip this time if everything is filled
        # identify the left and right nearest valid side

        if t == T - 1:  # if right end, use the previous one
            value = working_point_ref[t - 1, to_fill_mask].clone()
        else:
            # identify the right end, the left end must be filled in already
            to_fill_valid_curve = working_track_mask[t + 1 :, to_fill_mask]  # T,M
            # find the left most True slot
            to_fill_valid_curve = (
                to_fill_valid_curve.float() * inverse_muti[t + 1 :, None]
            )
            max_value, max_ind = to_fill_valid_curve.max(dim=0)
            # for no right mask case, use the left
            select_from = working_point_ref[t + 1 :, to_fill_mask]
            valid_right_end = torch.gather(
                select_from, 0, max_ind[None, :, None].expand(-1, -1, 3)
            )[
                0, max_value > 0
            ]  # valid when max_value > 0
            if t == 0:
                assert (
                    len(valid_right_end) == to_fill_valid_curve.shape[1]
                ), "empty noodle!"
                value = valid_right_end
            else:
                # must have a left end
                value = working_point_ref[t - 1, to_fill_mask].clone()
                valid_left_end = value[max_value > 0]
                delta_t = (
                    max_ind[max_value > 0] + 2
                )  # left valid, current, [0] in the max_ind
                delta_x = valid_right_end - valid_left_end
                inc = 1.0 * delta_x / delta_t[:, None]
                value[max_value > 0] = valid_left_end + inc
        working_point_ref[t, to_fill_mask] = value.clone()
    # np.savetxt("./debug/line_segment_init.xyz", point_ref.reshape(-1, 3).cpu().numpy())
    ret = point_ref.clone()
    ret[:, working_mask] = working_point_ref
    return ret.detach().clone()

def slot_o3d_outlier_identifyication(
    curve_xyz, curve_mask, nb_neighbors=20, std_ratio=2.0
):
    # curve_xyz: T,N,3, tensor

    assert curve_xyz.ndim == 3
    T, N, _ = curve_xyz.shape
    ret_inlier_mask = np.ones((T, N)) < 0  # all false
    for t in tqdm(range(T)):
        if not curve_mask[t].any():
            continue
        fg_mask = curve_mask[t].cpu()
        fg_xyz = curve_xyz[t].cpu().numpy()[fg_mask]
        inlier_mask_buffer = np.zeros(len(fg_xyz)) > 0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(fg_xyz)

        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        inlier_ind = np.asarray(ind)
        if len(inlier_ind) > 0:
            inlier_mask_buffer[inlier_ind] = True  # len(fg_xyz)
        _t_inlier_mask = ret_inlier_mask[t]
        _t_inlier_mask[fg_mask] = inlier_mask_buffer
        ret_inlier_mask[t] = _t_inlier_mask
    ret_inlier_mask = torch.from_numpy(ret_inlier_mask).bool().to(curve_xyz.device)
    logging.warning(
        f"O3D outlier has {ret_inlier_mask.sum()/curve_mask.sum()*100:.2f}% inliers ({ret_inlier_mask.sum()} inliers out of {curve_mask.sum()})"
    )
    return ret_inlier_mask

def curve_shaking_identification(curve_xyz, shacking_th=0.2):
    # the queried curve
    # T,N,3
    ref = (curve_xyz[2:] + curve_xyz[:-2]) / 2.0
    ref = torch.cat([curve_xyz[1:2], ref, curve_xyz[-2:-1]], 0)
    diff = (curve_xyz - ref).norm(dim=-1)
    inlier = diff < shacking_th
    return inlier
