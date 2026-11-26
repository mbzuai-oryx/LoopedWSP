import sys,json,re
for f in sys.argv[1:]:
    print("==", f)
    for l in open(f):
        if not re.match(r'^[A-Z][0-9]', l): continue
        n,j=l.split(' ',1); d=json.loads(j)
        print(d['step'], int(d['time']), 'onestep',d.get('onestep'), 'test',d['test']['solved'],'snap',d['test_snap']['solved'], 'firstErr',d['test']['median_first_err'], ('x2 %s gen %s gensnap %s'%(d['test_x2']['solved'],d['gen']['solved'],d['gen_snap']['solved'])) if 'gen' in d else '')
