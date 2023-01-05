# 🧗‍♂️ Shooting for the Moon

> MoonBoard Climbing Route Classification

## Getting Started

### Prerequisites

Before you continue, ensure you have installed the following requirements:
- [Python 3](https://www.python.org/)

### Setup

1. Create a virtualenv and activate it
    ```sh
    $ python3 -m venv venv
    
    # MacOS & Linux
    $ source venv/bin/activate
    
    # Windows
    $ venv\Scripts\activate
    ```
2. Install all packages
    ```sh
    $ python3 -m pip install -r requirements.txt
    ```

### Building the Dataset

Build the dataset by running the following commands at the root of the project.

```sh
$ cd pre_processing
$ python3 ./build_dataset.py
```

### Running the Models

From the root of the project, run either a single model with the following command

```sh
$ python3 ./models/<model>.py

# Example
$ python3 ./models/dummy_classifier_model.py
```

or run all models sequentially with the following command

```sh
$ ./run_models
```

After each model runs there will be an output of the results to the terminal and a CSV saved in the `./results` directory with the model name containing all the different values set for each hyperparameter.

## Data Collection and Pre-Processing

- [Moonboard Dataset Generator](https://github.com/AlessandroAvi/Moonboard_datasetGenerator) for automating the [MoonBoard app](https://www.moonboard.com/moonboard-app) to grab screenshots of the climbs and covert them to a parsable JSON format
- [Kaggle: MoonBoard](https://www.kaggle.com/datasets/eddous/moonboard?select=datasets) used by a script to convert JSON to 11px x 18px (width x height) images (i.e. 1 px per hold on the MoonBoard (this step shouldn't be needed))
- Convert each image to np array shape 18,11 (rows, cols) where
  - 0 is no hold
  - 1 is a hold
  - 2 is a start hold
  - 3 is a finish hold

  Store this list of arrays in a text file `routes.txt` with their corresponding grades from the JSON file in another txt file `grades.txt`
- [MoonBoard ML](https://github.com/mshr-h/moonboard-ml) used for going from raw `.txt` files to np array (including splitting data for training and testing)

## Team

| Name                | Student ID |
|---------------------|------------|
| Alexandru Bara      | 40132235   |
| Andre Ibrahim       | 40132881   |
| Domenic Seccareccia | 40063021   |
| Jason Gerard        | 40079266   |
| Jeremy Piperni      | 40177789   |