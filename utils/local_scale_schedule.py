"""Optional cap curriculum and per-group LR decay for fixed local-scale runs.

The cache describes the final allowed support. Only its current ceiling changes;
centers, learned scales, soft targets, and Adam moments are never inflated by a
schedule update. The policy and current iteration travel with the checkpoint.
"""
import json
import math


GROUP_LRS = dict(f_dc=.0025, f_rest=.000125, opacity=.01, scaling=.001, rotation=.0005)


def validate(policy, final_multiplier, iterations):
    if policy is None:
        return None
    if (not isinstance(policy, dict) or set(policy) != {'schema', 'cap', 'learning_rates'} or
            policy['schema'] != 1 or not (policy['cap'] or policy['learning_rates'])):
        raise ValueError('Invalid local-scale schedule policy')
    cap = policy['cap']
    if cap is not None:
        if not isinstance(cap, dict) or set(cap) != {'initial_multiplier', 'final_multiplier', 'start', 'end'}:
            raise ValueError('Invalid cap schedule fields')
        if (not all(isinstance(cap[k], (int, float)) and math.isfinite(cap[k]) for k in ('initial_multiplier', 'final_multiplier')) or
                not 0 < cap['initial_multiplier'] <= cap['final_multiplier'] or cap['final_multiplier'] != final_multiplier):
            raise ValueError('Cap schedule must increase to the cached final ceiling multiplier')
        validate_steps(cap, iterations)
    rates = policy['learning_rates']
    if rates is not None:
        if not isinstance(rates, dict) or set(rates) != {'start', 'end', 'groups'}:
            raise ValueError('Invalid learning-rate schedule fields')
        validate_steps(rates, iterations)
        if not isinstance(rates['groups'], dict) or set(rates['groups']) != set(GROUP_LRS):
            raise ValueError('Schedule all five non-position parameter groups explicitly')
        for group, endpoints in rates['groups'].items():
            if (not isinstance(endpoints, (list, tuple)) or len(endpoints) != 2 or
                    not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in endpoints) or
                    endpoints[1] > endpoints[0]):
                raise ValueError(f'Need positive, non-increasing LR endpoints for {group}')
    return policy


def validate_steps(schedule, iterations):
    if (not all(type(schedule[k]) is int for k in ('start', 'end')) or
            not 0 <= schedule['start'] < schedule['end'] <= iterations):
        raise ValueError('Schedule needs 0 <= start < end <= training iterations')


def progress(iteration, start, end):
    t = min(1., max(0., (iteration-start)/(end-start)))
    return .5-.5*math.cos(math.pi*t)


def values(policy, iteration):
    result = {}
    if policy['cap'] is not None:
        c = policy['cap']
        p = progress(iteration, c['start'], c['end'])
        result['ceiling_multiplier'] = c['initial_multiplier'] + p*(c['final_multiplier']-c['initial_multiplier'])
    if policy['learning_rates'] is not None:
        r = policy['learning_rates']
        p = progress(iteration, r['start'], r['end'])
        result['lrs'] = {name: (a if p == 0 else b if p == 1 else math.exp((1-p)*math.log(a)+p*math.log(b)))
                         for name, (a, b) in r['groups'].items()}
    return result


def bind(model, opt, *, fresh):
    """Called after optimizer construction and before initial scale projection."""
    text = getattr(opt, 'local_scale_schedule', '')
    policy = validate(json.loads(text) if text else None,
                      getattr(opt, 'local_scale_ceiling_multiplier', 1.), opt.iterations)
    state = model.fixed_lidar_state
    if policy is not None and state.get('scale_control') != 'hybrid':
        raise ValueError('Local-scale schedules require fixed LiDAR hybrid scale control')
    saved = state.get('local_scale_schedule')
    if not fresh and (saved['policy'] if saved else None) != policy:
        raise ValueError('Checkpoint local-scale schedule differs from requested schedule')
    if policy is None:
        return
    if fresh:
        state['local_scale_schedule'] = dict(policy=policy, iteration=0)
        if policy['cap'] is not None:
            state['log_scale_cap_final'] = state['log_scale_cap'].clone()
    elif (type(saved.get('iteration')) is not int or not 0 <= saved['iteration'] <= opt.iterations or
          (policy['cap'] is not None and 'log_scale_cap_final' not in state)):
        raise ValueError('Checkpoint is missing local-scale schedule state')
    update(model, state['local_scale_schedule']['iteration'])


def update(model, iteration):
    state = model.fixed_lidar_state
    scheduled = state.get('local_scale_schedule') if state is not None else None
    if scheduled is None:
        return
    if iteration < scheduled['iteration']:
        raise ValueError('Local-scale curriculum cannot rewind an existing training state')
    current = values(scheduled['policy'], iteration)
    if 'ceiling_multiplier' in current:
        multiplier = current['ceiling_multiplier']
        if multiplier != scheduled.get('ceiling_multiplier'):
            import torch
            final = scheduled['policy']['cap']['final_multiplier']
            # Recompute from immutable final bounds, without cumulative drift.
            with torch.no_grad():
                torch.add(state['log_scale_cap_final'], math.log(multiplier/final), out=state['log_scale_cap'])
            scheduled['ceiling_multiplier'] = multiplier
    if 'lrs' in current:
        for group in model.optimizer.param_groups:
            if group['name'] in current['lrs']:
                group['lr'] = current['lrs'][group['name']]
    scheduled['iteration'] = iteration


def stats(model):
    state = model.fixed_lidar_state
    scheduled = state.get('local_scale_schedule') if state is not None else None
    if scheduled is None:
        return {}
    result = dict(schedule_iteration=scheduled['iteration'],
                  current_ceiling_multiplier=scheduled.get('ceiling_multiplier', state.get('local_scale_ceiling_multiplier', 1.)))
    result.update({'lr_'+g['name']: g['lr'] for g in model.optimizer.param_groups})
    return result
