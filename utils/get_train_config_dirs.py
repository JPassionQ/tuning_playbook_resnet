import os

def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_paths.append(os.path.join(root, filename))
    return file_paths

if __name__ == "__main__":
    dir_path = "/home/jingqi/DeepLearningWorkshop/recipes/research_on_activation/round1"
    files = get_all_file_paths(dir_path)
    with open("/home/jingqi/DeepLearningWorkshop/recipes/research_on_activation/research_on_activation_recipes.txt", "w") as f:
        for file in files:
            f.write(file + "\n")