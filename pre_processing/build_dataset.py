import json
import os
import glob
import numpy as np
from PIL import Image

from grade_converter import grade_converter
from moonboard_utils import mb_coord_to_cartesian, moonboard_route_filter
from dataset_utils import save_dataset, load_dataset


# Rows and columns of the moonboard
ROWS = 18
COLUMNS = 11

# Colors to set the different mmonboard holds to in the intermediate representation
START_HOLD_COLOR = (0, 255, 0)
MIDDLE_HOLD_COLOR = (0, 0, 255)
END_HOLD_COLOR = (255, 0, 0)

# Map to convert the intermediate color values to single but values for the numpy array
color_to_bit = {
    (0, 0, 0): 0,
    MIDDLE_HOLD_COLOR: 1,
    START_HOLD_COLOR: 2,
    END_HOLD_COLOR: 3,
}


# After the images are pulled from the app and turned into a json representation we will then convert them back into
# a simplified image format that is 11 pixels wide by 18 pixels tall, 1 pixel for each possible hold on the moonboard
def convert_json_to_images(json_file_name, output_image_folder, max_number=None):
    json_routes = {}
    with open(json_file_name) as file:
        json_routes = json.load(file)

    print(f"Number of pre-filtered routes: {len(json_routes)}")

    # Filter out any routes that don't follow the specifications
    # Method -> "Feet follow hands"
    # MoonBoardHoldSetup -> "MoonBoard Masters 2017"
    # MoonboardConfiguration -> "40° MoonBoard"
    filtered_json_routes = []
    for route in json_routes:
        if moonboard_route_filter(route):
            filtered_json_routes.append(route)

    if max_number is not None:
        print(f"Cutting down dataset to {max_number} elements")
        filtered_json_routes = filtered_json_routes[:max_number]

    print(f"Number of filtered routes: {len(filtered_json_routes)}")

    # Next we will create a map of grade -> list of routes
    # where each route or moves is a tuple of lists of holds (start_holds, middle_holds, end_holds)
    grades_to_routes = {}
    for route in filtered_json_routes:
        grade = grade_converter(route["Grade"])
        if grade not in grades_to_routes:
            grades_to_routes[grade] = []

        raw_moves = route["Moves"]
        start = []
        middle = []
        end = []
        # Each move is labeled with booleans that says if it is a start, end, or middle climbing hold
        # so we separate them into 3 different lists here so we can color them differently which will
        # hopefully help the model
        for move in raw_moves:
            position = move["Position"]
            if move['IsStart']:
                start.append(position)
            elif move['IsEnd']:
                end.append(position)
            else:
                middle.append(position)

        grades_to_routes[grade].append((start, middle, end))

    for grade, routes in grades_to_routes.items():
        print(f"{len(routes)} number of routes for grade {grade}")

    # Filter out the samples associated to classes that have under 1,000 samples, this is the minimum threshold we set
    # in order to have balanced data. We also truncate the maximum number of samples to 10,000
    min = 1_000
    max = 10_000
    normalized_grades_to_routes = {}
    for grade, routes in grades_to_routes.items():
        if len(routes) < min:
            print(
                f"Skipping routes for grade {grade} because {len(routes)} is under of the target range [{min}, {max}]")
            continue

        if len(routes) > max:
            normalized_grades_to_routes[grade] = grades_to_routes[grade][:max]
        else:
            normalized_grades_to_routes[grade] = grades_to_routes[grade]

    print("Number of routes after normalization")
    for grade, routes in normalized_grades_to_routes.items():
        print(f"{len(routes)} number of routes for grade {grade}")

    # Clear the folder where the images are stored
    for file_name in os.listdir(output_image_folder):
        file = os.path.join(output_image_folder, file_name)
        if os.path.isfile(file) and ".png" in file_name:
            os.remove(file)

    # Loop through all the routes we have after filtering and save intermediate image representation for each
    for grade, routes in normalized_grades_to_routes.items():
        for i, route in enumerate(routes):
            # Create a new image
            route_image = Image.new("RGB", (COLUMNS, ROWS))

            moves_with_colors = zip(route, (START_HOLD_COLOR, MIDDLE_HOLD_COLOR, END_HOLD_COLOR))
            for move_coordinates, color in moves_with_colors:
                for coordinate in move_coordinates:
                    # Convert each coordinate into cartesian and then write the pixel with that color in the image
                    x, y = mb_coord_to_cartesian(coordinate)
                    route_image.putpixel((x, y), color)

            route_image.save(os.path.join(output_image_folder, f"{str(grade)}_{str(i)}.png"))


# The second phase of building the dataset includes converting those intermediate images into an even more simplified
# numpy array representation where each image is a 18x11 2D numpy array where each element can have 1 of 4 values each
# representing what kind of hold it is: unused, middle, start, end
def convert_images_to_npz(image_folder, npz_file_name):
    routes = []
    grades = []

    for file_name in os.listdir(image_folder):
        file = os.path.join(image_folder, file_name)
        if os.path.isfile(file) and ".png" in file_name:
            image = Image.open(file)
            # The intermediate representation uses an RGB image, so we account for that in the shape
            flat_pixels = np.array(image).reshape(11 * 18, 3)

            flat_bit_image = []
            # Ideally this loop can be vectorized to work faster with larger datasets
            for pixel in flat_pixels:
                flat_bit_image.append(color_to_bit[tuple(pixel)])

            bit_image = np.array(flat_bit_image).reshape(18, 11)
            assert bit_image.shape == (18, 11), f"Image {file_name} is not the correct shape"

            # Extract the route grade / difficulty / class / label from the filename
            grade = int(file_name.split("_")[0][1:])

            routes.append(bit_image)
            grades.append(grade)

    # After we build the lists for the routes and grades we will split the data into train and test sets
    assert len(grades) == len(routes), "The number of routes and grades are not equal"
    save_dataset(npz_file_name, routes, grades)


if __name__ == "__main__":
    # Setting the numpy random seed here ensure reproducibility of the dataset
    np.random.seed(42)

    # These are the 2 main steps for the pre-processing and dataset building pipeline, this assumes there is the initial
    # json file containing the data taken from automating the moonboard application
    convert_json_to_images(os.path.join("intermediate_json", "data.json"), os.path.join("intermediate_images"))
    convert_images_to_npz("intermediate_images", "dataset.npz")
