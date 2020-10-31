# perses

Assess damage of buildings.

## Table of Contents

-   [Getting Started](#getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Initial setup](#initial-setup)
-   [Usage](#usage)
-   [Contributing](#contributing)
-   [Versioning](#versioning)
-   [Authors](#authors)
-   [License](#license)
-   [Acknowledgments](#acknowledgments)

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes.

### Prerequisites

You will need python3 and pip3 installed on your machine. You can install it
from the official website https://www.python.org/.

To install pytorch with CUDA support, conda is recommended. An installation
guide is available in the conda docs:
https://docs.conda.io/projects/conda/en/latest/user-guide/install/

### Initial setup

A step by step series of examples that tell you how to get the project up and
running.

Clone the git repository

```bash
git clone https://github.com/intelligenerator/perses.git
cd perses
git submodule init
git config submodule.recurse true
git submodule update --init --recursive
```

Then create your conda virtual environment

```bash
conda create --name torch
conda activate torch
```

Next, installed the required packages. This may vary based on your system
hardware and requirements. Read more about pytorch installation:
https://pytorch.org/get-started/locally/

```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

To exit the virtual environment run

```bash
conda deactivate
```

Happy coding!

## Usage

Open the [dnet.ipynb](dnet.ipynb) notebook in jupyter lab to start experimenting
or to train your own model.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details on our code of conduct, and
the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available,
see the [tags on this repository](https://github.com/intelligenerator/unet_dataset/tags).

## Authors

Boldizsar Zopcsak - [BoldizsarZopcsak](https://github.com/BoldizsarZopcsak)

See also the list of
[contributors](https://github.com/intelligenerator/unet_dataset/contributors)
who participated in this project.

## License

This project is licensed under the MIT License - see the
[LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

-   [DEVELOPING CUSTOM PYTORCH DATALOADERS](https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html)
-   [Contributor Covenant](https://www.contributor-covenant.org/) - Code of Conduct
