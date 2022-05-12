# Image-Similarity
A Streamlit app that lets you compare image similarity via different optional metrics depending on how you want to measure similarity.

Play around with it here: https://share.streamlit.io/gabriel0110/image-similarity/main

## About

I wanted to build a solution for comparing image similarity, possibly to use in the future for building a recommendation system for images, e.g. image search. After research of some metrics, I put together this application to showcase the similarity metrics.

Currently, there three optional metrics:
- Cosine Similarity
- NAED (Neighbor-Average Euclidean Distance)
- SSIM (Structural Similarity Index)

***Disclaimer**: NAED is not an official/formal metric for image similarity -- it is something that I came up with and tested, and it appears to be a fairly plausible similarity metric for images that I wanted to include as an option.*

There is also the option to find and plot feature keypoint matches between the two images to include in your analysis of similarity if you wanted to.

## Findings on metric analysis

- NAED seems to be robust to image rotation, whereas L2 norm (Euclidean Distance) is not - likely because no pixels are changing color value in the rotation.
- The lower the NAED score, the most similar
- SSIM is not as robust to rotation as NAED, so be aware when comparing a target image augmented by rotation - SSIM will be poor, so it may not be best to use to include in overall similarity percentage if rotation does not matter for you.
- Cosine similarity seems to be the best OVERALL similarity factor, so it's probably best to always have this option selected to include in overall similarity calculation.
