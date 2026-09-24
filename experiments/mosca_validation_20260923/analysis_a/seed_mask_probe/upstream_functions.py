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

def get_dynamic_curves(
    s2d,
    cams: MonocularCameras,
    return_all_curves=False,
    # filter of 2D tracks to avoid the fg-bg error track flickering
    refilter_2d_track_flag=True,
    refilter_2d_track_only_mask=False,  # if set true won't remove any curve, just mark the outlier as invalid
    refilter_min_valid_cnt=2,
    refilter_o3d_nb_neighbors=16,
    refilter_o3d_std_ratio=5.0,
    refilter_shaking_th=0.2,
    refilter_remove_shaking_curve=False,  # if true, any curve with shaking will be totally removed
    refilter_spatracker_consistency_th=0.2,
    #
    spatracker_original_curve=False,
    # additional mask, this is for the semantic label consistency mask
    fg_additional_mask=None,
    # spatracker 3D curve also have a choice to use line init
    enforce_line_init=False,
    # subsample t list
    t_list=None,
    # safe cfgs
    min_num_curves=0,
):
    device = s2d.rgb.device
    # * load track
    if return_all_curves:
        track = s2d.track.clone()
        track_mask = s2d.track_mask.clone()
    else:
        track = s2d.track[:, s2d.dynamic_track_mask].clone()
        track_mask = s2d.track_mask[:, s2d.dynamic_track_mask].clone()
    if fg_additional_mask is not None:
        assert fg_additional_mask.shape == track_mask.shape
        track_mask = track_mask * fg_additional_mask

    filter_mask = torch.ones(track.shape[1]).bool().to(device)
    if t_list is None:
        t_list = torch.arange(s2d.T).to(device)  # use all
    else:
        t_list = torch.as_tensor(t_list).to(device)

    track = track[t_list]
    track_mask = track_mask[t_list]
    # ! the subsample may lead to all empty track, which will be filtered out later by the min valid cnt!

    if track.shape[-1] == 3:
        logging.info(f"SpaT mode, direct use 3D Track")
        # spa tracker model
        # manually homo list
        homo_list = __int2homo_coord__(track[..., :2], s2d.H, s2d.W)
        dep_list = track[..., -1]
        node_xyz_spatracker = get_world_points(homo_list, dep_list, cams, t_list)

        # todo: align the spatrack curve to depth
        # 1. get inlier mask by checking the unproject points
        # 2. for inliers, aign the curve depth by interpolating the per-frame valid alignment

        gathered_homo_list, gathered_dep_list, rgb_list = prepare_track_buffers(
            s2d, track[..., :2], track_mask, t_list
        )
        node_xyz_unproject = get_world_points(
            gathered_homo_list, gathered_dep_list, cams, t_list
        )
        # for visible slot, use gathered_curve else use
        if spatracker_original_curve:
            curve_xyz = node_xyz_spatracker
        else:
            mix_mask = track_mask.float()[..., None].expand(-1, -1, 3)
            curve_xyz = node_xyz_unproject * mix_mask + node_xyz_spatracker * (
                1.0 - mix_mask
            )

        # * 3D curve still need to use filter to exclud wrong fetch
        if refilter_2d_track_flag and not return_all_curves:
            inlier_mask = slot_o3d_outlier_identifyication(
                curve_xyz,
                track_mask,
                nb_neighbors=refilter_o3d_nb_neighbors,
                std_ratio=refilter_o3d_std_ratio,
            )
            logging.info(
                f"O3d outlier ratio {(~inlier_mask).float().mean()*100.0:.2f}%"
            )
            track_mask = track_mask * inlier_mask

            inlier_mask = curve_shaking_identification(curve_xyz, refilter_shaking_th)
            logging.info(
                f"shaking outlier ratio {(~inlier_mask).float().mean()*100.0:.2f}%"
            )
            if refilter_remove_shaking_curve:
                has_shaking = track_mask * (~inlier_mask)  # valid but has outlier
                has_shaking = has_shaking.any(0, keepdim=True)
                logging.info(f"{has_shaking.sum()} curves has shaking, remove")
                inlier_mask = (~has_shaking).expand(len(track_mask), -1)
            track_mask = track_mask * inlier_mask

            fetch_spatracker_diff = (node_xyz_unproject - node_xyz_spatracker).norm(
                dim=-1
            )
            spatracker_consistent_mask = (
                fetch_spatracker_diff < refilter_spatracker_consistency_th
            )
            logging.info(
                f"spatracker consistency newly detect {(~spatracker_consistent_mask & track_mask).float().mean()*100.0:.2f}% with th={refilter_spatracker_consistency_th}"
            )
            track_mask = track_mask * spatracker_consistent_mask

            if not refilter_2d_track_only_mask:
                # find valid cnt
                valid_cnt = track_mask.sum(0)
                if filter_mask.sum() > min_num_curves:
                    filter_mask = (valid_cnt >= refilter_min_valid_cnt) & filter_mask
                    logging.info(
                        f"Refiltering 2D tracks, {(~filter_mask).sum()} curves has less than {refilter_min_valid_cnt} valid slots, remove"
                    )
                track = track[:, filter_mask]
                track_mask = track_mask[:, filter_mask]
                # * redo
                homo_list = __int2homo_coord__(track[..., :2], s2d.H, s2d.W)
                dep_list = track[..., -1]
                node_xyz_spatracker = get_world_points(
                    homo_list, dep_list, cams, t_list
                )
                gathered_homo_list, gathered_dep_list, rgb_list = prepare_track_buffers(
                    s2d, track[..., :2], track_mask, t_list
                )
                node_xyz_unproject = get_world_points(
                    gathered_homo_list, gathered_dep_list, cams, t_list
                )
                # for visible slot, use gathered_curve else use
                mix_mask = track_mask.float()[..., None].expand(-1, -1, 3)
                if spatracker_original_curve:
                    curve_xyz = node_xyz_spatracker
                else:
                    curve_xyz = node_xyz_unproject * mix_mask + node_xyz_spatracker * (
                        1.0 - mix_mask
                    )
            else:
                # if use non-original 3D curve, need to re-mix the curve to remove the noise samling
                if not spatracker_original_curve:
                    mix_mask = track_mask.float()[..., None].expand(-1, -1, 3)
                    curve_xyz = node_xyz_unproject * mix_mask + node_xyz_spatracker * (
                        1.0 - mix_mask
                    )
        if enforce_line_init:
            curve_xyz = line_segment_init(track_mask, curve_xyz)

    else:
        logging.info(f"2D track mode, use line segment to fill")
        homo_list, dep_list, rgb_list = prepare_track_buffers(
            s2d, track[..., :2], track_mask, t_list
        )
        curve_xyz = line_segment_init(
            track_mask, get_world_points(homo_list, dep_list, cams, t_list).clone()
        )
        if refilter_2d_track_flag and not return_all_curves:
            inlier_mask = slot_o3d_outlier_identifyication(
                curve_xyz,
                track_mask,
                nb_neighbors=refilter_o3d_nb_neighbors,
                std_ratio=refilter_o3d_std_ratio,
            )
            track_mask = track_mask * inlier_mask

            inlier_mask = curve_shaking_identification(curve_xyz, refilter_shaking_th)
            logging.info(
                f"shaking outlier ratio {(~inlier_mask).float().mean()*100.0:.2f}%"
            )
            if refilter_remove_shaking_curve:
                has_shaking = track_mask * (~inlier_mask)  # valid but has outlier
                has_shaking = has_shaking.any(0, keepdim=True)
                logging.info(f"{has_shaking.sum()} curves has shaking, remove")
                inlier_mask = (~has_shaking).expand(len(track_mask), -1)

            track_mask = track_mask * inlier_mask

            if not refilter_2d_track_only_mask:
                # find valid cnt
                valid_cnt = track_mask.sum(0)
                if filter_mask.sum() > min_num_curves:
                    filter_mask = (valid_cnt >= refilter_min_valid_cnt) & filter_mask
                    logging.info(
                        f"Refiltering 2D tracks, {(~filter_mask).sum()} curves has less than {refilter_min_valid_cnt} valid slots, remove"
                    )
                track = track[:, filter_mask]
                track_mask = track_mask[:, filter_mask]
                # * redo
                homo_list, dep_list, rgb_list = prepare_track_buffers(
                    s2d, track[..., :2], track_mask, t_list
                )
                curve_xyz = line_segment_init(
                    track_mask,
                    get_world_points(homo_list, dep_list, cams, t_list).clone(),
                )
            else:
                raise NotImplementedError()

    return curve_xyz, track_mask, rgb_list, filter_mask
