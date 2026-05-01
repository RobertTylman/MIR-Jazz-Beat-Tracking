import json
import sys

def update_notebook(path, old_str, new_str):
    with open(path, 'r') as f:
        data = json.load(f)
    
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                new_source.append(line.replace(old_str, new_str))
            cell['source'] = new_source
            
    with open(path, 'w') as f:
        json.dump(data, f, indent=1)

if __name__ == "__main__":
    notebook_path = sys.argv[1]
    old = sys.argv[2]
    new = sys.argv[3]
    update_notebook(notebook_path, old, new)
