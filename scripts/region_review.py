"""Persistent manual decisions over automatic regional camera candidates."""
import json
from pathlib import Path
import uuid


DECISIONS = ('unreviewed', 'keep', 'reject')


def reviewed_rows(rows, decisions):
    names = [row['name'] for row in rows]
    if len(names) != len(set(names)) or set(decisions) != set(names):
        raise ValueError('Review must contain exactly the current camera candidates')
    if any(value not in DECISIONS for value in decisions.values()):
        raise ValueError('Invalid camera review decision')
    return [row for row in rows if decisions[row['name']] == 'keep']


class CameraReview:
    def __init__(self, identity, config, rows, decisions=None, current=None):
        self.identity, self.config, self.rows = identity, config, rows
        self.names = [row['name'] for row in rows]
        if not self.names:
            raise ValueError('No camera candidates to review')
        self.decisions = dict(decisions) if decisions is not None else dict.fromkeys(self.names, 'unreviewed')
        reviewed_rows(rows, self.decisions)
        self.current = current if current in self.names else self.names[0]

    def choose(self, name, decision):
        if name not in self.decisions or decision not in DECISIONS:
            raise ValueError('Unknown camera or review decision')
        self.decisions[name] = decision

    def filtered(self, decision='all'):
        if decision not in ('all', *DECISIONS):
            raise ValueError('Unknown review filter')
        return [name for name in self.names if decision == 'all' or self.decisions[name] == decision]

    def neighbor(self, name, direction=1, decision='all'):
        available = set(self.filtered(decision))
        if not available:
            return None
        start = self.names.index(name) if name in self.names else (-1 if direction > 0 else 0)
        for offset in range(1, len(self.names)+1):
            candidate = self.names[(start+direction*offset) % len(self.names)]
            if candidate in available:
                return candidate

    def counts(self):
        kept = reviewed_rows(self.rows, self.decisions)
        train = sum(row['split'] == 'train' for row in kept)
        return dict(candidates=len(self.names), kept=len(kept), train=train, heldout=len(kept)-train,
                    rejected=sum(v == 'reject' for v in self.decisions.values()),
                    unreviewed=sum(v == 'unreviewed' for v in self.decisions.values()))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            temporary.write_text(json.dumps(dict(schema=1, identity=self.identity, config=self.config,
                candidates=self.rows, decisions=self.decisions, current=self.current),indent=2,allow_nan=False)+'\n')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path, identity):
        value = json.loads(Path(path).read_text())
        if value.get('schema') != 1 or value.get('identity') != identity:
            raise ValueError('Saved review uses different source inputs/sampling; choose a new --output or --review-file')
        return cls(identity,value['config'],value['candidates'],value['decisions'],value.get('current'))
