# DHH23 Portfolio

This is a repository for my individual portfolio of the [Helsinki Digital Humanities Hackathon 23](https://www.helsinki.fi/en/digital-humanities/helsinki-digital-humanities-hackathon-2023-dhh23). 
The Early Modern group explored reuse in scientific illustrations in the 18th century.
As a continuation of the group's work "Illustrating the Enlightenment - Illuminating the Evolution of Scientific Illustration during the 18th century" (see our [poster](https://www.helsinki.fi/assets/drupal/2023-06/dhh23-earlymodern-poster.pdf)), 
in this portfolio, I further explored the reuse of images. First, I evaluated the performance of the developed reuse detection algorithm and then identified and discussed some influential reuse cases found in the data.

## Workflow

1) Computing reuse
- The reuse algorithm was adapted from [Aleksi Suuronen](https://github.com/AlluSu/image-similarity-detection/blob/main/code/similarities.py) and [Ari Vesalainen](https://github.com/vesalaia/Image_similarity)
- A short comparison of the resnet18 and resnet50 models.
	- Selecting the model and setting the final threshold for the similarity scores.

2) Exploring reuse through identifying influential works / authors
- Finding the image ids and publication ids that have been reused the most
- For this purpose, the data was filtered so that exact reprints of the same work were excluded, but reprinting in different formats was kept, since I was interested to see how pictures “travelled” across different publications.
Identifying the most reused authors and/or publications (the most influential?)

3) A case study on Linné's classification system
- (Where) have Linné’s pictures (“exact” matches by the similarity algorithm) been used by other authors?


## Data

Data used in this portfolio is from [Early Modern sample](https://github.com/dhh23/early_modern_data#early_modern_data)

## References

TBA