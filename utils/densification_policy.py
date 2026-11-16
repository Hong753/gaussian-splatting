"""Independent opacity-reset switch for runs with native densification."""


def opacity_reset_due(opt, phase_iteration, white_background=False, soft_surface=False):
    # Called inside the active native densification phase by train.py.
    if getattr(opt, 'disable_opacity_resets', False) or soft_surface:
        return False
    return (phase_iteration % opt.opacity_reset_interval == 0 or
            (white_background and phase_iteration == opt.densify_from_iter))
