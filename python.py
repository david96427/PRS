import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random

# ========== CÀI ĐẶT GIAO DIỆN CHUNG ==========
st.set_page_config(page_title="Recommendation System", layout="wide")
st.title("Recommendation System: ")
st.markdown("<h1 style='color:black; font-size: 30px;'> <b>Thời Trang Nam.</b></h1>", unsafe_allow_html=True)

menu = ["Home", "Project Introduction", "Achievements", "Users"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.markdown("---")
st.sidebar.image("shoppee_menu.jpg", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("👩‍🏫 **Giảng viên:**")
st.sidebar.info("Cô: Khuất Thùy Phương")
st.sidebar.markdown("🎖️ **Thực hiện bởi:**")
st.sidebar.info("Dương Đại Dũng")
st.sidebar.info("Nguyễn Thị Cẩm Thu")
st.sidebar.markdown("📅 **Ngày báo cáo:** 13/04/2025")

# ========== HÀM HIỂN THỊ SẢN PHẨM ==========
def display_recommended_products(recommended_products, cols=3):
    for i in range(0, len(recommended_products), cols):
        columns = st.columns(cols)
        for j, col in enumerate(columns):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                with col:
                    if isinstance(product['image'], str) and product['image']:
                        try:
                            st.image(product['image'], caption=product['product_name'], use_container_width=True)
                        except Exception as e:
                            st.warning(f"Không thể hiển thị ảnh cho sản phẩm '{product['product_name']}': {e}")

                    # Hiển thị tên và mã sản phẩm
                    st.markdown(f"**{product['product_name']}**")
                    st.write(f"Mã sản phẩm: `{product['product_id']}`")  # 👉 THÊM DÒNG NÀY

                    # Hiển thị giá tiền nếu có
                    if pd.notna(product.get('price', None)):
                        try:
                            rounded_price = round(float(product['price']))
                            st.write(f"**Giá:** {rounded_price:,.0f} VNĐ")
                        except:
                            pass

                    # Mô tả sản phẩm
                    exp = st.expander(" Mô tả")
                    truncated = ' '.join(str(product.get('Content', '')).split()[:30]) + "..."
                    exp.write(truncated)



# ========== TRANG CHỦ ==========
if choice == 'Home':  
    st.subheader("[Trang chủ](https://shopee.vn/)")
    st.image('shoppee4.jpg', use_container_width=True)
    st.write('ĐỒ ÁN TỐT NGHIỆP: DL07_DATN_K302 ')

# ========== GIỚI THIỆU DỰ ÁN ==========
elif choice == 'Project Introduction':    
    st.image('shoppee1.jpg', use_container_width=True)
    st.markdown("### [Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)")
    st.write("""
        - Trong bối cảnh thị trường thời trang nam trực tuyến trên Shopee ngày càng phát triển, việc giúp người dùng dễ dàng khám phá những sản phẩm phù hợp với sở thích và phong cách cá nhân trở nên vô cùng quan trọng. Việc xây dựng một hệ thống gợi ý thông minh, nhằm giải quyết bài toán về sự quá tải thông tin và nâng cao trải nghiệm mua sắm thời trang nam cho người dùng Shopee.
        - Sử dụng hai phương pháp:
            - **Collaborative Filtering:**
            - **Content-Based Filtering:**
    """)

# ========== THÀNH TỰU ==========
elif choice == 'Achievements': 
    st.image('CF_CBF.jpg', use_container_width=True)
    st.write('Collaborative Filtering:')
    st.image('SVD.jpg', use_container_width=True)
    st.write('==> Chọn model SVD của Surprise là tối ưu nhất.')
    st.write('Content-Based Filtering:')
    st.image('gensim_cosine.jpg', use_container_width=True)
    st.write("""
        - Xin phép sử dụng cosine similarity.
        - Xây hệ thống gợi ý tất cả sản phẩm --> nhiều người dùng: gensim (TfidfModel, Similarity)
    """)

# ========== NGƯỜI DÙNG ==========
elif choice == 'Users':
    st.image('shoppee2.jpg', use_container_width=True)

    # ===== LOAD DỮ LIỆU CẦN THIẾT =====
    with open('products_df.pkl', 'rb') as f:
        df_products = pickle.load(f)
    with open('recommendations_dict.pkl', 'rb') as f:
        recommendations_dict = pickle.load(f)
    with open("svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)
    with open("sample_df.pkl", "rb") as f:
        ratings_df = pickle.load(f)
    with open("products_name_df.pkl", "rb") as f:
        products_name_df = pickle.load(f)

    # ===== GỢI Ý CONTENT-BASED =====


    st.markdown("##  Gợi ý theo sản phẩm bạn đã chọn")

    # Thay thế selectbox bằng text input để người dùng nhập từ khóa
    keyword = st.text_input("Nhập từ khóa sản phẩm (ví dụ: áo khoác)")

    # Lọc sản phẩm dựa trên từ khóa
    if keyword:
        filtered_products = df_products[df_products['product_name'].str.contains(keyword, case=False, na=False)]

        if not filtered_products.empty:
            # Ưu tiên sản phẩm có hình ảnh
            products_with_image = filtered_products[filtered_products['image'].notna() & (filtered_products['image'] != '')]
            products_without_image = filtered_products[filtered_products['image'].isna() | (filtered_products['image'] == '')]

            # Lấy tối đa 20 sản phẩm, ưu tiên có hình ảnh
            top_n = 20
            suggested_products = pd.concat([products_with_image.head(top_n),
                                            products_without_image]).head(top_n)

            if not suggested_products.empty:
                # Hiển thị các sản phẩm đã lọc để người dùng chọn
                product_options = [(row['product_name'], row['product_id']) for _, row in suggested_products.iterrows()]
                selected_product = st.selectbox("Chọn sản phẩm từ gợi ý", options=product_options, format_func=lambda x: x[0])

                if selected_product:
                    selected_name = selected_product[0]
                    selected_ma_sp = selected_product[1]

                    st.write("Tên sản phẩm đã chọn:", selected_name)
                    st.write("Mã sản phẩm:", selected_ma_sp)

                    selected_info = df_products[df_products['product_id'] == selected_ma_sp]

                    if not selected_info.empty:
                        product_row = selected_info.iloc[0]
                        product_name = product_row['product_name']
                        product_image = product_row['image']
                        product_content = product_row['Content']
                        product_price = product_row['price']

                        st.markdown(f"### {product_name}")

                        if pd.notna(product_image) and product_image != "":
                            st.image(product_image, caption=product_name, use_container_width=True)

                        truncated_desc = ' '.join(str(product_content).split()[:50]) + "..."
                        st.write("**Mô tả:**", truncated_desc)

                        if pd.notna(product_price):
                            st.write("**Giá tiền:**", f"{round(product_price):,} VNĐ")

                        st.markdown("### Sản phẩm liên quan bạn có thể thích:")
                        related_ids = recommendations_dict.get(selected_ma_sp, [])
                        # Lọc các sản phẩm tương tự
                        related_products = df_products[df_products['product_id'].isin(related_ids)]

                        # Ưu tiên sản phẩm có ảnh
                        with_images_related = related_products[related_products['image'].notna() & (related_products['image'] != '')]
                        without_images_related = related_products[related_products['image'].isna() | (related_products['image'] == '')]

                        # Kết hợp đủ 9 sản phẩm: ưu tiên ảnh, thiếu thì bổ sung không ảnh
                        combined_products = pd.concat([with_images_related, without_images_related]).head(9)

                        # Hiển thị gợi ý
                        display_recommended_products(combined_products, cols=3)

            else:
                st.warning("Không tìm thấy sản phẩm nào phù hợp với từ khóa này.")

        else:
            st.warning("Không tìm thấy sản phẩm nào phù hợp với từ khóa này.")

    else:
        st.info("Vui lòng nhập từ khóa để tìm kiếm sản phẩm.")

    st.image('shoppee3.jpg', use_container_width=True)




    # ===== GỢI Ý THEO USER =====

    st.markdown("---")
    st.markdown("## 🔀 Gợi ý hybrid: Kết hợp hành vi & nội dung")

    # Gợi ý user có nhiều đánh giá
    top_active_users = ratings_df['user_id'].value_counts().head(10).index.tolist()
    st.write('Một số user hoạt động nhiều:', top_active_users)

    user_input = st.text_input("Nhập mã người dùng (user_id):")

    if user_input:
        try:
            user_id = int(user_input)
            rated_products = ratings_df[ratings_df["user_id"] == user_id]["product_id"].tolist()
            all_products = df_products["product_id"].tolist()
            unrated_products = [pid for pid in all_products if pid not in rated_products]

            # Nếu user chưa từng đánh giá
            if len(rated_products) == 0:
                st.warning("Người dùng chưa có lịch sử đánh giá. Đang sử dụng hệ thống content-based.")
                random_pid = random.choice(df_products["product_id"].tolist())
                content_ids = recommendations_dict.get(random_pid, [])[:9]
                hybrid_df = df_products[df_products["product_id"].isin(content_ids)]
            else:
                # --- Collaborative filtering ---
                predictions = []
                for pid in unrated_products:
                    pred = svd_model.predict(user_id, pid)
                    predictions.append((pid, pred.est))

                top_preds_cf = sorted(predictions, key=lambda x: x[1], reverse=True)
                top_cf_ids = [pid for pid, _ in top_preds_cf[:20]]  # lấy nhiều hơn 9

                # --- Content-based từ các sản phẩm đã rating ---
                content_ids = []
                for pid in rated_products:
                    content_ids.extend(recommendations_dict.get(pid, []))
                content_ids = list(set(content_ids))  # bỏ trùng

                # --- Trộn điểm gợi ý với trọng số ---
                hybrid_score = {}
                for i, pid in enumerate(top_cf_ids):
                    hybrid_score[pid] = hybrid_score.get(pid, 0) + (20 - i) * 1.0  # collaborative trọng số 1.0

                for i, pid in enumerate(content_ids):
                    hybrid_score[pid] = hybrid_score.get(pid, 0) + (20 - i) * 0.5  # content trọng số 0.5

                # --- Loại sản phẩm đã đánh giá
                for pid in rated_products:
                    hybrid_score.pop(pid, None)

                # --- Sắp xếp theo score và lấy top 9
                sorted_hybrid_ids = sorted(hybrid_score.items(), key=lambda x: x[1], reverse=True)
                final_ids = [pid for pid, _ in sorted_hybrid_ids][:9]

                hybrid_df = df_products[df_products["product_id"].isin(final_ids)]

            # Ưu tiên sản phẩm có ảnh
            with_images = hybrid_df[hybrid_df['image'].notna() & (hybrid_df['image'] != '')]
            without_images = hybrid_df[~hybrid_df['product_id'].isin(with_images['product_id'])]
            combined_df = pd.concat([with_images, without_images]).head(9)

            st.markdown("###  Gợi ý dành cho bạn:")
            display_recommended_products(combined_df, cols=3)

        except ValueError:
            st.warning("Vui lòng nhập `user_id` hợp lệ (số nguyên).")







