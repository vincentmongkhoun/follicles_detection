# Detection of ovarian follicles


Authors : François Caud and Alexandre Gramfort (Université Paris-Saclay)


## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Challenge description

Get started with the [dedicated notebook]


### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)


## Notes for development


### repository structure

Folders `data` and `models` are not committed

```
% tree -L 3 .
.
├── OD.ipynb
├── OpenSans-Regular.ttf
├── README.md
├── clf.ipynb
├── crop_images.ipynb
├── data
│   ├── coupes_jpg
│   │   └── D-1M01-2.jpg
│   ├── coupes_tiff
│   │   ├── D-1M01-2.tiff
│   │   ├── D-1M01-3.tiff
│   │   ├── D-1M01-4.tiff
│   │   ├── D-1M01-5.tiff
│   │   └── D-1M02-4.tiff
│   ├── scenario1
│   │   ├── Images
│   │   ├── Masks
│   │   ├── Xml
│   │   ├── labels.csv
│   │   └── split
│   └── scenario2
│       ├── Images
│       ├── Masks
│       └── Xml
├── models
│   └── classifier
│       ├── assets
│       ├── keras_metadata.pb
│       ├── saved_model.pb
│       └── variables
├── preprocessing_scripts
│   └── xml_to_csv.py
├── requirements.txt
├── retina.ipynb
├── sliding_window.ipynb
└── split_data_train_test.py
```

### Pre-processing raw data


- 1. Convert source .tiff images to lighter .jpg

    ```
    # TODO
    ```

- 2. Convert xml files of labels and bounding boxes to a single `labels.csv`

    ```
    % python preprocessing_scripts/xml_to_csv.py data/scenario1/Xml 
    Reading xml files in data/scenario1
    Successfully converted xml to csv: data/scenario1/labels.csv
    ```

    