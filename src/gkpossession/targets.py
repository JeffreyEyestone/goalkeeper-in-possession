"""Conservative *candidate* intended receipt reconstruction for audit only.

UUID relation and same-team/period/recipient agreement are required. This function
never substitutes a realized pass endpoint. Receipt locations remain retrospective
annotations of intent, not observed player locations at release.
"""
def candidate_receipt(event, by_id):
    recipient=event.get('pass',{}).get('recipient',{}).get('id')
    found=[]
    for eid in event.get('related_events',[]):
        e=by_id.get(eid,{})
        if (e.get('type',{}).get('name')=='Ball Receipt*'
            and e.get('team',{}).get('id')==event.get('team',{}).get('id')
            and e.get('period')==event.get('period')
            and recipient is not None and e.get('player',{}).get('id')==recipient
            and len(e.get('location',[]))>=2): found.append(e)
    return found[0] if len(found)==1 else None
