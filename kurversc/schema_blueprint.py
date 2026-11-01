"""Bind a frozen relational program to its schema-derived source instances.

The candidate occurrence limit prunes a source superset. It cannot construct
missing source paths. Source bounds and alias bindings therefore belong to
the learned program's replay contract, independently of its chosen topology.
No data values, label statistics or benchmark identities are used here.
"""
from collections import Counter


PLAN_KEY='kurversc_source_blueprint'


def _columns(value):return [value] if isinstance(value,str) else list(value or ())


def occurrence_bound(*,configs=None,options=None,default=1):
    values=[default]
    if configs is not None:values.extend(config.max_base_table_occurrences for config in configs)
    if options is not None:values.extend(options)
    if any(isinstance(value,bool) or not isinstance(value,int) or value<1 for value in values):
        raise ValueError('Source occurrence bounds must be positive integers')
    return max(values)


def source_blueprint(tables,relationships,root):
    if root not in tables:raise ValueError('The source blueprint must contain its predictive root')
    records=[];depths=[];occurrences=[]
    for name,table in tables.items():
        bounds=getattr(table,'path_source_bounds',None)
        if bounds is not None:
            depths.append(bounds[0]);occurrences.append(bounds[1])
        path=tuple(table.path_base_tables or ())
        if path:
            depths.append(len(path)-1)
            occurrences.append(max(Counter(path).values()))
        records.append(dict(name=name,prefix=table.prefix,key=_columns(table.key),
            path=list(path),date=table.date,available_at=table.available_at,
            cutoff_boundary=table.cutoff_boundary,
            context_keys=list(table.context_keys),model_exclusions=list(table.model_exclusions)))
    edges=[dict(parent=edge.parent,child=edge.child,parent_key=_columns(edge.parent_key),
        child_key=_columns(edge.child_key),reduce=edge.reduce) for edge in relationships]
    return dict(version=1,root=root,maximum_depth=max(depths) if depths else None,
        max_base_table_occurrences=max(occurrences) if occurrences else None,
        instances=records,relationships=edges)


def replay_limits(plan,*,depth,occurrences):
    blueprint=(plan or {}).get(PLAN_KEY)
    if blueprint is None:return depth,occurrences
    if blueprint.get('version')!=1:raise ValueError('Unsupported source-blueprint contract')
    return (max(1,blueprint['maximum_depth']) if blueprint['maximum_depth'] is not None else depth,
        blueprint['max_base_table_occurrences'] if blueprint['max_base_table_occurrences'] is not None else occurrences)


def verify_replay(plan,actual):
    expected=(plan or {}).get(PLAN_KEY)
    if expected is not None and expected!=actual:
        raise ValueError('Frozen source blueprint differs from the supplied source-instance bindings')
