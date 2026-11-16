"""Independent opacity-reset switch for runs with native densification."""


def opacity_reset_due(opt, phase_iteration, white_background=False, soft_surface=False):
    # Called inside the active native densification phase by train.py.
    if getattr(opt, 'disable_opacity_resets', False) or soft_surface:
        return False
    return (phase_iteration % opt.opacity_reset_interval == 0 or
            (white_background and phase_iteration == opt.densify_from_iter))


def size_pruning_thresholds(opt, phase_iteration, extent):
    """Screen radius and world sigma thresholds have independent switches."""
    start = getattr(opt, 'prune_size_from_iter', -1)
    if start < 0:
        start = opt.opacity_reset_interval
    if phase_iteration <= start:
        return None, None
    return (getattr(opt, 'prune_screen_size', 20.),
            getattr(opt, 'prune_world_scale_ratio', .1) * extent)
