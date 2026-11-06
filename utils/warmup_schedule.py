"""Position scheduling with a true fixed-position warmup and a linear ramp."""

def position_lr_step(iteration, warmup_steps, ramp_steps):
    if iteration <= warmup_steps:
        return 0.0, 0
    elapsed = iteration - warmup_steps
    if ramp_steps and elapsed <= ramp_steps:
        return elapsed / ramp_steps, 0
    return 1.0, max(0, elapsed - ramp_steps)


def densification_phase(iteration, offset):
    return iteration - offset
