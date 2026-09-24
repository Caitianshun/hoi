def check_smpl_exists():
    import os
    candidates = [
        f'{CACHE_DIR_4DHUMANS}/data/smpl/SMPL_NEUTRAL.pkl',
        f'data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
    ]
    candidates_exist = [os.path.exists(c) for c in candidates]
    if not any(candidates_exist):
        raise FileNotFoundError(f"SMPL model not found. Please download it from https://smplify.is.tue.mpg.de/ and place it at {candidates[1]}")

    # Code edxpects SMPL model at CACHE_DIR_4DHUMANS/data/smpl/SMPL_NEUTRAL.pkl. Copy there if needed
    if (not candidates_exist[0]) and candidates_exist[1]:
        convert_pkl(candidates[1], candidates[0])

    return True
