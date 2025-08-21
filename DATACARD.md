# Dataset Card for SynthHuman

This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 3000 samples.

## Installation

If you haven't already, install FiftyOne:

```bash
pip install -U fiftyone
```

## Usage

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load the dataset
# Note: other available arguments include 'max_samples', etc
dataset = load_from_hub("Voxel51/SynthHuman")

# Launch the App
session = fo.launch_app(dataset)
```


## Dataset Details

### Dataset Description

The SynthHuman dataset is a high-fidelity synthetic dataset created for training human-centric computer vision models. It contains 300,000 high-resolution (384×512) images with three main tasks: relative depth estimation, surface normal estimation, and soft foreground segmentation. The dataset features procedurally-generated human subjects in diverse poses, environments, lighting, and appearances, with equal distribution of faces, upper body, and full body scenarios. 

Unlike scan-based synthetic datasets, SynthHuman uses high-fidelity procedural generation techniques to create detailed human representations, including strand-level hair (with hundreds of thousands of individual 3D strands per hairstyle), detailed clothing, accessories, and expressive faces. This approach enables the generation of ground-truth annotations with strand-level granularity that capture fine details like facial wrinkles, eyelids, hair strands, and subtle texture variations.

- **Curated by:** Microsoft Research, Cambridge, UK

- **Funded by:** Microsoft

- **Shared by:** Microsoft

- **Language(s) (NLP):** en

- **License:** [CDLA - Permissive - 2.0](https://github.com/microsoft/DAViD/blob/main/LICENSE-CDLA-2.0.txt)

### Dataset Sources

- **Repository:** https://aka.ms/DAViD
  
- **Paper:** DAViD: Data-efficient and Accurate Vision Models from Synthetic Data (arXiv:2507.15365)

- **Parsing to FiftyOne format:** https://github.com/harpreetsahota204/synthhuman_to_fiftyone

## Uses

### Direct Use

The SynthHuman dataset is designed for training computer vision models for human-centric dense prediction tasks, specifically:

1. **Relative depth estimation**: Predicting per-pixel depth values for human subjects

2. **Surface normal estimation**: Predicting per-pixel surface normal vectors (xyz components)
 
3. **Soft foreground segmentation**: Generating soft alpha masks to separate humans from backgrounds

The dataset enables training smaller, more efficient models that achieve state-of-the-art accuracy without requiring large-scale pretraining or complex multi-stage training pipelines. This makes it suitable for applications with computational constraints.

### Out-of-Scope Use

The dataset should not be used for:
- Identifying or recognizing specific individuals
- Creating deceptive or misleading synthetic human content
- Applications that could violate privacy or cause harm to real individuals
- Training models for tasks beyond the three specified dense prediction tasks without proper evaluation

## Dataset Structure

The SynthHuman dataset contains 300,000 synthetic images of resolution 384×512, with equal distribution (100,000 each) across three categories:
1. Face scenarios
2. Upper body scenarios
3. Full body scenarios

Each sample in the dataset includes:
- RGB rendered image
- Soft foreground mask (alpha channel)
- Surface normals (3-channel)
- Depth ground-truth annotations

The dataset is designed to be diverse in terms of:
- Human poses and expressions
- Environments and lighting conditions
- Physical appearances (body shapes, clothing, accessories)
- Camera viewpoints

## Dataset Creation

### Curation Rationale

The dataset was created to address limitations in existing human-centric computer vision datasets, which often suffer from:
1. Imperfect ground truth annotations due to reliance on photogrammetry or noisy sensors
2. Limited diversity in subjects and environments due to challenges in capturing in-the-wild data
3. Inability to capture fine details like hair strands, reflective surfaces, and subtle geometric features

### Source Data

#### Data Collection and Processing

The dataset generation process involved sampling from:

- Face/body shapes (from training sources and a library of 3572 scans)
  
- Expressions and poses (from AMASS, MANO, and other sources)

- Textures (from high-resolution face scans with expression-based dynamic wrinkle maps)

- Hair styles (548 strand-level 3D hair models, each with 100K+ strands)

- Accessories (36 glasses, 57 headwear items)

- Clothing (50+ clothing tops)

- Environments (mix of HDRIs and 3D environments)

Rendering the complete dataset took 72 hours on a cluster of 300 machines with M60 GPUs (equivalent to 2 weeks on a 4-GPU A100 machine).

#### Who are the source data producers?

The dataset was created by researchers at Microsoft Research in Cambridge, UK. The synthetic data was procedurally generated using artist-created assets, scanned data sources, and procedural generation techniques.

### Annotations

#### Annotation process

Since this is a synthetic dataset, the annotations are generated programmatically during the rendering process rather than being manually created. This ensures perfect alignment between the RGB images and their corresponding ground truths.

Special attention was given to:
1. **Hair representation**: A voxel-grid volume with density based on strand geometry was created, then converted to a coarse proxy mesh using marching cubes to generate interpretable normal vectors.
2. **Transparent surfaces**: The dataset provides control over whether depth and normals of translucent surfaces (like eyeglass lenses) are included or whether they show the surface behind them.
3. **Soft foreground masks**: Generated with pixel-perfect accuracy, including partial transparency for hair strands and other fine structures.

#### Personal and Sensitive Information

The dataset contains only synthetic human representations and does not include any real personal or sensitive information. The synthetic data generation process ensures that no real individuals are represented in the dataset.

## Bias, Risks, and Limitations

While the paper doesn't explicitly discuss biases in the dataset, there are potential limitations:
- The paper notes that there may be aspects of human diversity not yet represented in the dataset
- The synthetic nature of the data might not fully capture all real-world scenarios and edge cases
- Models trained on this data may have lower accuracy for some demographic groups (acknowledged as a potential issue in the paper)
- Failure cases noted in the paper include extreme lighting conditions, printed patterns on clothing, tattoos, and rare scale variations (e.g., a baby held in an adult's hand)

### Recommendations

Users should be aware of the following:
- The dataset creators acknowledge that the synthetic data approach helps address fairness concerns by providing precise control over the training data distribution
- Additional diversity in assets and scene variations could improve robustness to real-world scenarios
- Users should test models trained on this data across diverse real-world populations to ensure fair performance
- For applications involving human subjects, users should consider the ethical implications and potential biases
- Supplementing with real-world data for specific challenging scenarios might be beneficial

## Citation

**BibTeX:**

```bibtex
@misc{saleh2025david,
    title={{DAViD}: Data-efficient and Accurate Vision Models from Synthetic Data},
    author={Fatemeh Saleh and Sadegh Aliakbarian and Charlie Hewitt and Lohit Petikam and Xiao-Xian and Antonio Criminisi and Thomas J. Cashman and Tadas Baltrušaitis},
    year={2025},
    eprint={2507.15365},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2507.15365},
}
```

**APA:**
Saleh, F., Aliakbarian, S., Hewitt, C., Petikam, L., Criminisi, A., Cashman, T. J., & Baltrusaitis, T. (2025). DAViD: Data-efficient and Accurate Vision Models from Synthetic Data. arXiv preprint arXiv:2507.15365.