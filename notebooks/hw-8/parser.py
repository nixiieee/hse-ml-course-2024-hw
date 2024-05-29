import os
import pandas as pd

ext_match = {
    '.py': 'PYTHON', '.cpp': 'CPLUSPLUS', '.js': 'JAVASCRIPT', 
    '.java': 'JAVA', '.md': 'MARKDOWN', '.ps1': 'POWERSHELL', 
    '.kt': 'KOTLIN', '.hs': 'HASKELL', '.yml': 'YAML', '.c': 'C',
    '.h': 'C', '.hpp': 'CPLUSPLUS'
}

def find_files_with_extensions(directory, extensions):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_list.append(os.path.join(root, file))
    return file_list

def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except:
        return None

def get_top_level_subdirectory(file_path, base_directory):
    relative_path = os.path.relpath(file_path, base_directory)
    top_level_dir = relative_path.split(os.sep)[0]
    return top_level_dir

def create_dataframe_from_files(directory, extensions):
    files = find_files_with_extensions(directory, extensions)
    data = {'text': [], 'language': [], 'top_level_dir': []}
    
    for file_path in files:
        content = read_file(file_path)
        if content is not None:
            ext = os.path.splitext(file_path)[1]
            data['text'].append(content)
            data['language'].append(ext_match[ext])
            data['top_level_dir'].append(get_top_level_subdirectory(file_path, directory))
    
    df = pd.DataFrame(data)
    return df

directory = '../../data/programming_languages'
extensions = ['.cpp', '.py', '.js', '.java', '.c', '.hs', '.md', '.yml', '.ps1', '.kt', '.h', '.hpp'] 
df = create_dataframe_from_files(directory, extensions)
df.to_csv('../../data/github_code_medium.csv')
