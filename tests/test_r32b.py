from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_r32b_taxonomy_and_proxy_definitions_are_locked():
 s=(ROOT/'scripts/r32b.py').read_text()
 for role in ['CB','FB_WB','DM_6','CM_8','AM_10','WINGER','STRIKER','OTHER_UNKNOWN']: assert role in s
 assert "receiver_role.eq('DM_6')" in s and "receiver_role.eq('STRIKER')" in s
 assert 'target_distance_m' in s and 'realized_distance_m' not in s[s.index("x['central_6_proxy']"):]
