"""YAML-driven Patient model builder.

The Patient Pydantic model is auto-generated at import time from
``docs/parameters_schema.yaml``.  Every YAML entry with a ``disorders:``
key (even an empty list) is exposed as a Patient field — that key is the
single source of truth for "which parameters the LLM can tune."

Each generated field gets:
    - default       — pulled from PARAMS_DEFAULT.{class}.{field}
    - description   — built from YAML description + disorder examples
    - constraints   — ge/le/min_length/max_length from YAML range/length

Aliases (parameters that map to a slice or transform of a real model field,
e.g. ``b_vs_L`` writing into ``brain.b_vs[3:6]``) are handled via the
explicit ALIASES table below — the auto-builder skips fields it can't find
in PARAMS_DEFAULT and the alias entry provides type/default.

Public API:
    Patient                  — auto-generated Pydantic class
    apply_patient(p, params) — copy patient values onto Params, return new Params
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any
from copy import deepcopy

import jax.numpy as jnp
import yaml
from pydantic import BaseModel, Field, create_model
from pydantic.functional_validators import AfterValidator

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, with_sensory, with_plant,
)


_YAML_PATH = Path(__file__).resolve().parent.parent.parent.parent / 'docs' / 'parameters_schema.yaml'


# ── Aliases: patient-facing names that don't map 1:1 to a model field ─────────
# Each alias has:
#   default     — patient-side default
#   apply(params, value) → new params  — how to write the patient value into model params

def _broadcast_b_vs(params):
    """b_vs may be a scalar (broadcasts) or a (6,) array — return as (6,)."""
    arr = jnp.asarray(params.brain.b_vs)
    if arr.ndim == 0:
        arr = jnp.full((6,), float(arr))
    return arr


def _apply_b_vs_L(params, value):
    # b_vs_L sets the LEFT VN pop slice — model layout: [right pop (3), left pop (3)]
    # So slots [3:6] correspond to model RIGHT pop = anatomical LEFT VN.
    bvs = _broadcast_b_vs(params).at[3:6].set(float(value))
    return with_brain(params, b_vs=bvs)


def _apply_b_vs_R(params, value):
    bvs = _broadcast_b_vs(params).at[0:3].set(float(value))
    return with_brain(params, b_vs=bvs)


ALIASES = {
    'brain.b_vs_L': {
        'default': 100.0,
        'pytype': float,
        'apply':  _apply_b_vs_L,
    },
    'brain.b_vs_R': {
        'default': 100.0,
        'pytype': float,
        'apply':  _apply_b_vs_R,
    },
}


# ── YAML loading ──────────────────────────────────────────────────────────────

def _load_schema() -> dict:
    with open(_YAML_PATH, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


# ── Description builder ───────────────────────────────────────────────────────

def _format_disorders(disorders: list) -> str:
    if not disorders:
        return ''
    lines = ['', 'Clinical examples:']
    for d in disorders:
        name  = d.get('name', '?')
        value = d.get('value', '?')
        lines.append(f'  • {name}: {value}')
    return '\n'.join(lines)


def _build_description(entry: dict) -> str:
    desc = (entry.get('description') or '').strip()
    examples = _format_disorders(entry.get('disorders') or [])
    if examples:
        desc = f'{desc}\n{examples}'
    return desc


# ── Field type / constraint builder ───────────────────────────────────────────

def _make_list_validator(rng: list, length: int | None):
    """Return a per-element range validator for list fields."""
    lo, hi = rng if rng else (None, None)

    def _check(v):
        if length is not None and len(v) != length:
            raise ValueError(f'must be a list of length {length}, got {len(v)}')
        for i, x in enumerate(v):
            if lo is not None and x < lo:
                raise ValueError(f'index {i}: {x} < {lo}')
            if hi is not None and x > hi:
                raise ValueError(f'index {i}: {x} > {hi}')
        return v
    return _check


def _build_field(entry: dict, default: Any) -> tuple:
    """(field_type, Field(...)) for one YAML entry."""
    description = _build_description(entry)
    rng    = entry.get('range') or []
    length = entry.get('length')
    is_list = (length is not None) or hasattr(default, '__len__') and not isinstance(default, str)

    if is_list:
        # List parameter: per-element validator.
        if hasattr(default, 'tolist'):
            default_list = list(default.tolist())
        else:
            default_list = list(default)
        validator = _make_list_validator(rng, length)
        # Default Field constraints for length (Pydantic handles lengths natively).
        kwargs = {'description': description}
        if length is not None:
            kwargs['min_length'] = length
            kwargs['max_length'] = length
        field_type = Annotated[list[float], AfterValidator(validator)]
        return field_type, Field(default=default_list, **kwargs)

    # Scalar parameter: ge/le constraints.
    kwargs = {'description': description}
    if rng:
        lo, hi = rng[0], rng[1] if len(rng) > 1 else None
        if lo is not None:
            kwargs['ge'] = lo
        if hi is not None:
            kwargs['le'] = hi
    return float, Field(default=float(default), **kwargs)


# ── Resolve default from PARAMS_DEFAULT ───────────────────────────────────────

def _resolve_model_default(class_name: str, field_name: str):
    """Return default value from PARAMS_DEFAULT.{class}.{field}, or None if missing."""
    cls = getattr(PARAMS_DEFAULT, class_name, None)
    if cls is None:
        return None
    return getattr(cls, field_name, None)


# ── Build the Patient class ───────────────────────────────────────────────────

def _build_patient_class():
    schema = _load_schema()
    fields: dict[str, tuple] = {}
    field_to_class: dict[str, str] = {}    # patient field name → 'brain'/'sensory'/'plant'/'alias'

    for key, entry in schema.items():
        if not isinstance(entry, dict):
            continue
        if 'disorders' not in entry:
            continue   # not LLM-exposed
        try:
            class_name, field_name = key.split('.', 1)
        except ValueError:
            continue
        if class_name not in ('brain', 'sensory', 'plant'):
            continue
        if field_name in fields:
            continue   # already built (rare collision case)

        # Aliases: use ALIASES table for type/default
        if key in ALIASES:
            alias = ALIASES[key]
            default = alias['default']
            pytype  = alias['pytype']
            kwargs  = {'description': _build_description(entry)}
            rng     = entry.get('range') or []
            if rng:
                if rng[0] is not None: kwargs['ge'] = rng[0]
                if len(rng) > 1 and rng[1] is not None: kwargs['le'] = rng[1]
            fields[field_name] = (pytype, Field(default=default, **kwargs))
            field_to_class[field_name] = 'alias'
            continue

        # Real model field: pull default from PARAMS_DEFAULT
        default = _resolve_model_default(class_name, field_name)
        if default is None:
            continue   # YAML entry without a backing model field — skip silently
        ftype, fobj = _build_field(entry, default)
        fields[field_name] = (ftype, fobj)
        field_to_class[field_name] = class_name

    Patient = create_model('Patient', **fields, __base__=BaseModel)
    Patient.__doc__ = (
        'Model parameter overrides relative to healthy defaults. '
        'Auto-generated from docs/parameters_schema.yaml — every entry with '
        'a `disorders:` key is exposed as a field here.'
    )
    return Patient, field_to_class


Patient, _FIELD_CLASS_MAP = _build_patient_class()


# ── Apply patient → params ────────────────────────────────────────────────────

def apply_patient(patient: Patient, params=PARAMS_DEFAULT):
    """Apply all patient-set fields onto base params, returning a new Params."""
    p = params
    # Group updates per class so we hit with_brain/with_sensory/with_plant once each.
    brain_updates: dict[str, Any]   = {}
    sensory_updates: dict[str, Any] = {}
    plant_updates: dict[str, Any]   = {}
    for fname, klass in _FIELD_CLASS_MAP.items():
        value = getattr(patient, fname, None)
        if value is None:
            continue
        if klass == 'alias':
            full_key = next(k for k in ALIASES if k.endswith(f'.{fname}'))
            p = ALIASES[full_key]['apply'](p, value)
            continue
        # Real field — convert lists to jnp arrays
        default = _resolve_model_default(klass, fname)
        if hasattr(default, 'shape'):
            value = jnp.asarray(value, dtype=jnp.float32)
        target = {'brain': brain_updates, 'sensory': sensory_updates, 'plant': plant_updates}[klass]
        target[fname] = value
    if brain_updates:
        p = with_brain(p, **brain_updates)
    if sensory_updates:
        p = with_sensory(p, **sensory_updates)
    if plant_updates:
        p = with_plant(p, **plant_updates)
    return p
