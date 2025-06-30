import kagglehub
import pathlib, os

# Download latest version
path = kagglehub.dataset_download("kushsheth/netflix-eda-reccomendation-system")

print("Path to dataset files:", path)

directory_filenames = [i for i in os.listdir(path)]

new_directory = "C:/Users/kazmi/Documents/Kaggle_Projects/Netflix_Reccomendation System/Netflix_Recommendation_System/Data/"

print("Filenames: ", directory_filenames)

for i in directory_filenames:
    os.rename(
        os.path.join(path,i).replace("\\", '/'),
        new_directory + '\\' + i
        )
    


