import urllib.request, json

queries = ['wesad', 'deap emotion eeg', 'daisee engagement attention', 'stress physiological', 'rafdb facial']
for q in queries:
    url = 'https://huggingface.co/api/datasets?search=' + q.replace(' ', '+') + '&limit=5'
    try:
        r = urllib.request.urlopen(url, timeout=8)
        data = json.loads(r.read())
        hits = [(d.get('id'), d.get('downloads', 0)) for d in data]
        print('[' + q + ']', hits)
    except Exception as e:
        print('[' + q + '] ERR:', str(e))
