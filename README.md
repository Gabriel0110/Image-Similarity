# Image-Similarity
A Streamlit app that lets you compare image similarity via different optional metrics depending on how you want to measure similarity.

Play around with it here: https://share.streamlit.io/gabriel0110/image-similarity/main

## About

I wanted to build a solution for comparing image similarity, possibly to use in the future for building a recommendation system for images, e.g. image search. This project required me to research some metrics on image comparison and similarity, test, and play around with them. After research of some metrics, I put together this application to showcase them.

Currently, there are three optional metrics:
- Cosine Similarity
- NAED (Neighbor-Average Euclidean Distance)
- SSIM (Structural Similarity Index)

***Disclaimer**: NAED is not an official/formal metric for image similarity -- it is something that I came up with and tested, and it appears to be a fairly plausible similarity metric for images that I wanted to include as an option.*

The total similarity percentage score that is displayed after calculating the metrics is based on how many metrics you are using, e.g. 1/3 metrics == 100% (no split), 2/3 metrics == 50% split, 3/3 metrics == 33% split.

There is also the option to find and plot feature keypoint matches between the two images to include in your analysis of similarity if you wanted to. This uses RootSIFT, a variant of SIFT to find feature keypoints. My implementation is based off of Adrian Rosebrock's tutorial here: https://pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/

## Findings on metric analysis

- NAED seems to be robust to image rotation, whereas L2 norm (Euclidean Distance) is not - likely because no pixels are changing color value in the rotation.
- The lower the NAED score, the most similar
- SSIM is not as robust to rotation as NAED, so be aware when comparing a target image augmented by rotation - SSIM will be poor, so it may not be best to use to include in overall similarity percentage if rotation does not matter for you.
- Because SSIM is based on structural similarity, this score can be very low if the structural differences between the two images is high. For this reason, SSIM may not always be a desired option to include in total similarity calculation, especially if you do not care about structural similarity. Otherwise, SSIM is a good way to scrutinize similarity even further depending on how harsh you want to be on finding a similar image.
- Cosine similarity seems to be the ***best overall*** similarity factor, so it's probably best to always have this option selected to include in overall similarity calculation, with the option of scrutinizing the similarity score with the other two metrics.

## Demo video
https://user-images.githubusercontent.com/25640614/168156945-962c9a13-f097-44e3-8cb8-d72026d9a333.mp4
