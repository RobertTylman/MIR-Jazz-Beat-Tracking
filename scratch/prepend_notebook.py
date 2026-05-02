import json
import sys

def prepend_to_cell(path, cell_index, lines):
    with open(path, 'r') as f:
        data = json.load(f)
    
    cell = data['cells'][cell_index]
    cell['source'] = lines + cell['source']
            
    with open(path, 'w') as f:
        json.dump(data, f, indent=1)

if __name__ == "__main__":
    notebook_path = sys.argv[1]
    cell_idx = int(sys.argv[2])
    # The rest are lines to prepend
    lines_to_add = [l + '\n' for l in sys.argv[3:]]
    prepend_to_cell(notebook_path, cell_idx, lines_to_add)
