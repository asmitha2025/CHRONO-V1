import json
file_path = 'c:/Users/harih/OneDrive/Desktop/CHRONO-V1/notebooks/chrono_gemma4_demo.ipynb'
with open(file_path, 'r') as f:
    data = json.load(f)

for cell in data.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'model_id = "google/gemma-3-4b-it"' in line:
                new_source.append(line.replace('google/gemma-3-4b-it', 'google/gemma-4-e4b-it'))
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w') as f:
    json.dump(data, f, indent=1)
