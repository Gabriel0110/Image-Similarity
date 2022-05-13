import streamlit as st
from rootsift_driver import RootSiftDriver

from image_similarity import *

ref_img = None
target_img = None
pil_ref_img = None
pil_target_img = None

st.title("Image Similarity")
st.text("Determine similarity between two images based on multiple metrics.")
st.write("")
st.write("")

col1, col2 = st.columns(2)

uploaded_ref_image = col1.file_uploader("Choose a reference image to compare to...")
if uploaded_ref_image is not None:
    ref_img = get_opencv_image(uploaded_ref_image)
    pil_ref_img = get_pil_image(uploaded_ref_image)

    # Convert from BGR to RGB color space
    ref_img = cv2.cvtColor(np.float32(ref_img), cv2.COLOR_BGR2RGB)
    pil_ref_img = pil_ref_img.convert('RGB')

    col1.image(ref_img, caption='Reference image, resized to 512x512', use_column_width=True)
    st.write("")

uploaded_target_image = col2.file_uploader("Choose a target image for comparison...")
if uploaded_target_image is not None:
    target_img = get_opencv_image(uploaded_target_image)
    pil_target_img = get_pil_image(uploaded_target_image)

    # Convert from BGR to RGB color space
    target_img = cv2.cvtColor(np.float32(target_img), cv2.COLOR_BGR2RGB)
    pil_target_img = pil_target_img.convert('RGB')

    col2.image(target_img, caption='Target "compare-to" image, resized to 512x512', use_column_width=True)
    st.write("")

if uploaded_ref_image is not None and uploaded_target_image is not None:
    st.write("Select any metrics you wish to use for similarity (Note: will be included in total similarity score calculation):")
    cosine_check = st.checkbox("Cosine Similarity")
    naed_check = st.checkbox("NAED (Neighbor-Average Euclidean Distance)")
    ssim_check = st.checkbox("SSIM (Structural Similarity Index)")

    if cosine_check or naed_check or ssim_check:
        if st.button('Get Similarity Metrics'):
            # Determine the divisor for total percent calculation based on how many metrics are selected
            percent_proportion = 100.0 / sum([cosine_check, naed_check, ssim_check])
            total_percent = 0.0

            if naed_check:
                # Get euclidean distance between both neighbor-averaged images (requires grayscale images)
                ref_img_gray = cv2.cvtColor(np.float32(ref_img), cv2.COLOR_BGR2GRAY)
                target_img_gray = cv2.cvtColor(np.float32(target_img), cv2.COLOR_BGR2GRAY)
                l2_na_value = get_L2_norm_neighbor_avg(ref_img, target_img)
                l2_percent_of_img_size = l2_na_value / ((IMG_SIZE[0] * IMG_SIZE[1]) / 255) # smaller percentage is better
                normalized_l2_percent = 100 - (l2_percent_of_img_size * 100) # convert the small percentage to a large percentage equivalent
                #print(f"L2 norm of neighbor-avg: {l2_na_value:.6f} ({normalized_l2_percent:.2f}%)")

                naed_percent = (((((l2_percent_of_img_size) - L2_NA_THRESHOLD) * 100) / (0.0 - L2_NA_THRESHOLD)) * percent_proportion) / 100
                total_percent += naed_percent
                #print(naed_percent)

            if ssim_check:
                # Get SSIM value
                ssim_value = get_ssim(ref_img, target_img)
                #print(f"SSIM: {ssim_value:.6f}") # SSIM closest to 1 is the most similar

                ssim_percent = ((((ssim_value - 0.0) * 100) / (1.0 - 0.0)) * percent_proportion) / 100
                total_percent += ssim_percent
                #print(ssim_percent)

            if cosine_check:
                # Flatten to 1-D for cosine similarity
                ref_img_flattened = ref_img.flatten()
                target_img_flattened = target_img.flatten()
                cosine_sim_value = get_cosine_similarity(ref_img_flattened, target_img_flattened)
                #print(f"Cosine similarity: {cosine_sim_value:.6f}")

                cosine_sim_percent = ((((cosine_sim_value - 0.0) * 100) / (1.0 - 0.0)) * percent_proportion) / 100
                total_percent += cosine_sim_percent
                #print(cosine_sim_percent)

            if total_percent >= 90:
                st.markdown("<div style='text-align: center'>The two images are <b>VERY SIMILAR</b></div>", unsafe_allow_html=True)
                st.write("")
            elif total_percent < 90 and total_percent >= 80:
                st.markdown("<div style='text-align: center'>The two images are <b>SIMILAR</b></div>", unsafe_allow_html=True)
                st.write("")
            elif total_percent < 80 and total_percent >= 70:
                st.markdown("<div style='text-align: center'>The two images are <b>SLIGHTLY SIMILAR</b></div>", unsafe_allow_html=True)
                st.write("")
            elif total_percent < 70:
                st.markdown("<div style='text-align: center'>The two images are <b>NOT SIMILAR</b></div>", unsafe_allow_html=True)
                st.write("")

            st.metric("SIMILARITY SCORE:", f"{total_percent:.2f}%")
            st.write("")
            st.write("")

            col1, col2 = st.columns(2)
            with col1:
                if cosine_check:
                    st.metric("Cosine Similarity", f"{cosine_sim_value:.6f}")
                if naed_check:
                    st.metric("Neighbor-Average Euclidean Distance", f"{l2_na_value:.2f} ({normalized_l2_percent:.2f}%)")
                if ssim_check:
                    st.metric("SSIM", f"{ssim_value:.6f}")

            with col2:
                if cosine_check:
                    st.metric("Ideal Similarity Threshold", f">= {COSINE_THRESHOLD}")
                if naed_check:
                    st.metric("Ideal Similarity Threshold", f">= {(L2_NA_THRESHOLD*100) - 20}%")
                if ssim_check:
                    st.metric("Ideal Similarity Threshold", f">= {SSIM_THRESHOLD}")


            with st.expander("More about these metrics..."):
                if cosine_check:
                    st.write("""
                        Cosine similarity measures the similarity between two vectors of an inner product space. It is a common
                        technique for measuring similarity in text, but can also be used for images with the requirement of
                        flattening the image arrays to 1-D vectors. This metric seems to be the best OVERALL factor of similarity 
                        after testing many images and might be the only metric you need if structural similarity and color are not 
                        desired metrics when comparing. The closer to 1 the value, the more similar the images are.
                    """)
                    st.write("")
                if naed_check:
                    st.write("""
                        Neighbor-Average Euclidean Distance (NAED) is the distance between two images by their pixel values 
                        after converting each pixel value to the average of their surrounding neighbors. The smaller this value, 
                        the more similar the images are. It seems that color is most determining factor for this value. If there 
                        is a lot of similar color in the same locations of the two images, then this score will be higher. The 
                        percentage is the value as a percent of the total image size normalized by subtracting it from 100%. With 
                        the normalized percentage, the higher the value, the more similar the images.
                    """)
                    st.write("")
                if ssim_check:
                    st.write("""
                        SSIM attempts to measure structural similarity, unlike MSE (mean squared error) which attempts to measure
                        perceived errors. MSE can run into problems with image similarity, so SSIM is a solid choice to combat 
                        those problems and give you a better idea of how similar the images are. It is good to note that SSIM is not 
                        as robust to rotation as NAED. Rotation will make the SSIM score worse. With testing, it seems anything below 
                        0.35 is not similar in the slightest. The closer to 1, the similar the images are.
                    """)
                    st.write("")

            st.write("")
            st.write("")
            st.write("")

        if st.button("Find keypoints and similarity matches..."):
            rs_driver = RootSiftDriver()
            matches, max_matches, good_matches, kp_img1, kp_img2, feature_matches_img = rs_driver.run_sift(ref_img, target_img)

            col1, col2 = st.columns(2)
            with col1:
                col1.image(kp_img1, caption="Reference image with keypoint markers", use_column_width=True)
            with col2:
                col2.image(kp_img2, caption="Target image with keypoint markers", use_column_width=True)

            st.image(feature_matches_img, caption="Good/high-quality feature similarity matches between the two images' keypoints", use_column_width=True)
            st.write("")
            st.write(f"{len(matches)} keypoint matches were found out of {max_matches} possible matches ({(len(matches) / max_matches)*100:.2f}%)")
            st.write(f"Out of the {len(matches)} matches found, {len(good_matches)} ({(len(good_matches) / len(matches))*100:.2f}%) were good, high quality matches based on Lowe's ratio.")

st.write("")
st.write("")
st.write("")
st.write("")