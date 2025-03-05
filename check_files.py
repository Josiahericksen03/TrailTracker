import os

data_dir = "/Users/josiahericksen/PycharmProjects/Trailtracker/data/train"

animal_classes = ["Bear", "Turkey", "Boar", "Bobcat", "Deer", "Unidentifiable"]

def count_images(data_dir, animal_classes):
    total_images = 0
    for animal in animal_classes:
        animal_dir = os.path.join(data_dir, animal)
        num_images = len([f for f in os.listdir(animal_dir) if f.endswith(('jpg', 'jpeg', 'png'))])
        print(f"{animal}: {num_images} images")
        total_images += num_images
    print(f"Total images: {total_images}")

count_images(data_dir, animal_classes)
