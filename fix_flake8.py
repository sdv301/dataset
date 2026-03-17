import subprocess
import re

cmd = ["python", "-m", "flake8", r"c:\Users\pc24\Downloads\Code\dataset", "--select=F541,E722", "--format=default"]
result = subprocess.run(cmd, capture_output=True, text=True)

files_changed = {}

for line in result.stdout.splitlines():
    match = re.match(r"(.*?):(\d+):(\d+): (.*)", line)
    if match:
        filepath, row, col, msg = match.groups()
        row = int(row)
        
        if filepath not in files_changed:
            with open(filepath, 'r', encoding='utf-8') as f:
                files_changed[filepath] = f.readlines()
            
        lines = files_changed[filepath]
            
        if "F541" in msg:
            line_content = lines[row-1]
            # remove the `f` from the string
            line_content = re.sub(r'\bf(["\'])', r'\1', line_content)
            lines[row-1] = line_content
            
        elif "E722" in msg:
            line_content = lines[row-1]
            lines[row-1] = re.sub(r'except:', r'except Exception as e:', line_content)

for filepath, lines in files_changed.items():
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
        
print(f"Fixed {len(files_changed)} files.")
